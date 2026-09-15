from functools import lru_cache
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "IRIS ReID"
    host: str = "127.0.0.1"
    port: int = 8003
    frontend_port: int = 5176
    data_dir: Path = PROJECT_ROOT / ".data"
    frontend_dist: Path = PROJECT_ROOT / "frontend/dist"
    gpu_lock_path: Path = PROJECT_ROOT / ".runtime/gpu0.lock"
    reid_python: Path = PROJECT_ROOT / ".runtime/reid-env/Scripts/python.exe"
    reid_model_dir: Path | None = None
    max_upload_gb: float = 1.0
    model_config = SettingsConfigDict(env_prefix="IRIS_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    @model_validator(mode="after")
    def resolve_paths(self):
        for name in ("data_dir", "frontend_dist", "gpu_lock_path", "reid_python", "reid_model_dir"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, (PROJECT_ROOT / value).resolve())
        return self

    @property
    def database_path(self):
        return self.data_dir / "reid.db"

    def ensure_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings():
    settings = Settings()
    settings.ensure_directories()
    return settings
