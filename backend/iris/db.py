from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel


SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    kind TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (kind, id)
);
CREATE INDEX IF NOT EXISTS idx_records_kind_updated ON records(kind, updated_at DESC);
CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    """Small thread-safe JSON document repository backed by SQLite."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    @staticmethod
    def _json(value: BaseModel | dict[str, Any] | list[Any] | str | int | float | bool | None) -> str:
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        return json.dumps(value, default=str, separators=(",", ":"))

    def put(self, kind: str, record_id: str, value: BaseModel | dict[str, Any]) -> None:
        payload = self._json(value)
        with self._lock:
            self._connection.execute(
                """INSERT INTO records(kind, id, data) VALUES (?, ?, ?)
                   ON CONFLICT(kind, id) DO UPDATE SET data=excluded.data, updated_at=CURRENT_TIMESTAMP""",
                (kind, record_id, payload),
            )
            self._connection.commit()

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT data FROM records WHERE kind=? AND id=?", (kind, record_id)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def list(self, kind: str, *, limit: int = 500, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT data FROM records WHERE kind=? ORDER BY updated_at DESC, id LIMIT ? OFFSET ?", (kind, limit, offset)
            ).fetchall()
        return [json.loads(row["data"]) for row in rows]

    def delete(self, kind: str, record_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM records WHERE kind=? AND id=?", (kind, record_id)
            )
            self._connection.commit()
        return cursor.rowcount > 0

    def apply_changes(self, upserts, deletes, states=()):
        """Commit related document updates, removals, and counters together."""
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                for kind, value in upserts:
                    self._connection.execute(
                        "INSERT INTO records(kind,id,data) VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data,updated_at=CURRENT_TIMESTAMP",
                        (kind, value["id"], self._json(value)))
                self._connection.executemany("DELETE FROM records WHERE kind=? AND id=?", deletes)
                for key, value in states:
                    self._connection.execute(
                        "INSERT INTO app_state(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
                        (key, self._json(value)))
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def put_many(self, kind: str, values: Iterable[BaseModel]) -> None:
        rows = [(kind, str(item.id), self._json(item)) for item in values]  # type: ignore[attr-defined]
        with self._lock:
            self._connection.executemany(
                """INSERT INTO records(kind, id, data) VALUES (?, ?, ?)
                   ON CONFLICT(kind, id) DO UPDATE SET data=excluded.data, updated_at=CURRENT_TIMESTAMP""",
                rows,
            )
            self._connection.commit()

    def set_state(self, key: str, value: Any) -> None:
        with self._lock:
            self._connection.execute(
                """INSERT INTO app_state(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
                (key, self._json(value)),
            )
            self._connection.commit()

    def activate_model(self, current_id: str, previous_id: str, models: Iterable[BaseModel]) -> None:
        """Atomically update model flags and activation/rollback pointers."""
        rows = [("model", str(item.id), self._json(item)) for item in models]  # type: ignore[attr-defined]
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.executemany(
                    """INSERT INTO records(kind, id, data) VALUES (?, ?, ?)
                       ON CONFLICT(kind, id) DO UPDATE SET data=excluded.data, updated_at=CURRENT_TIMESTAMP""",
                    rows,
                )
                self._connection.executemany(
                    """INSERT INTO app_state(key, value) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
                    (("active_model", self._json(current_id)), ("previous_model", self._json(previous_id))),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._connection.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def close(self) -> None:
        with self._lock:
            self._connection.close()
