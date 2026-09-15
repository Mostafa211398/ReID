from contextlib import asynccontextmanager
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .db import Database
from .resources import ResourceCoordinator
from .reid_routes import reid_router
from .reid_service import ReIDService


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")[:160] or "crop.jpg"


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int):
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, "Upload exceeds configured size limit")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return size


class Services:
    def __init__(self, settings):
        settings.ensure_directories()
        self.settings = settings
        self.db = Database(settings.database_path)
        self.coordinator = ResourceCoordinator(settings.gpu_lock_path, "reid")
        self.reid = ReIDService(self.db, settings, self.coordinator)

    def close(self):
        self.reid.close()
        self.db.close()


def create_app(settings: Settings | None = None):
    settings = settings or get_settings()
    services = Services(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        services.close()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.services = services
    app.include_router(reid_router(services.reid, save_upload, safe_filename))

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def conflict(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/api/system/health")
    def health():
        return {"status": "ok", "app": settings.app_name}

    if settings.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
    return app
