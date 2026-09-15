"""Small shared utilities, with no model registry or media-service imports."""
import hashlib
from datetime import datetime, timezone

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def utc_now():
    return datetime.now(timezone.utc)


def sha256_file(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
