from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


class ResourceCoordinator:
    """Serializes GPU work for a single low-memory accelerator."""

    def __init__(self, lock_path: Path | None = None, app_id: str = "iris") -> None:
        self._condition = threading.Condition()
        self._owner: str | None = None
        self._priority_waiters = 0
        self._lock_path = lock_path.resolve() if lock_path else None
        self._owner_path = self._lock_path.with_suffix(".owner.json") if self._lock_path else None
        self._app_id = app_id
        self._global_handle: BinaryIO | None = None

    @property
    def owner(self) -> str | None:
        with self._condition:
            local = self._owner
        if local:
            return f"{self._app_id}:{local}"
        return self._external_owner()

    @staticmethod
    def _try_lock(handle: BinaryIO) -> bool:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            return False

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _open_lock(self) -> BinaryIO:
        assert self._lock_path is not None
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._lock_path.open("a+b")
        if self._lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        return handle

    def _external_owner(self) -> str | None:
        if not self._lock_path or not self._lock_path.exists():
            return None
        handle = self._open_lock()
        try:
            if self._try_lock(handle):
                self._unlock(handle)
                return None
        finally:
            handle.close()
        if self._owner_path and self._owner_path.exists():
            try:
                value = json.loads(self._owner_path.read_text(encoding="utf-8"))
                return f"{value.get('app', 'external')}:{value.get('task', 'gpu task')}"
            except Exception:
                pass
        return "external:gpu task"

    def _acquire_global(self, owner: str, cancel_event: threading.Event | None, wait: bool) -> bool:
        if not self._lock_path:
            return True
        handle = self._open_lock()
        while not self._try_lock(handle):
            if not wait:
                handle.close()
                return False
            if cancel_event is not None and cancel_event.is_set():
                handle.close()
                return False
            threading.Event().wait(0.25)
        self._global_handle = handle
        if self._owner_path:
            self._owner_path.write_text(json.dumps({"app": self._app_id, "task": owner, "pid": os.getpid(), "acquired_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        return True

    def acquire(
        self,
        owner: str,
        *,
        wait: bool = True,
        cancel_event: threading.Event | None = None,
        priority: bool = False,
    ) -> bool:
        with self._condition:
            if not wait and (self._owner is not None or (not priority and self._priority_waiters)):
                return False
            if priority:
                self._priority_waiters += 1
            try:
                while self._owner is not None or (not priority and self._priority_waiters):
                    if cancel_event is not None and cancel_event.is_set():
                        return False
                    self._condition.wait(timeout=0.25)
                if cancel_event is not None and cancel_event.is_set():
                    return False
                self._owner = owner
            finally:
                if priority:
                    self._priority_waiters -= 1
                    self._condition.notify_all()
        if self._acquire_global(owner, cancel_event, wait):
            return True
        with self._condition:
            if self._owner == owner:
                self._owner = None
                self._condition.notify_all()
        return False

    def release(self, owner: str) -> None:
        with self._condition:
            if self._owner == owner:
                if self._global_handle is not None:
                    try:
                        if self._owner_path:
                            try:
                                self._owner_path.unlink(missing_ok=True)
                            except OSError:
                                pass
                        self._unlock(self._global_handle)
                    finally:
                        self._global_handle.close()
                        self._global_handle = None
                self._owner = None
                self._condition.notify_all()
