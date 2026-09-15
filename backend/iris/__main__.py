import uvicorn
from .config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("iris.app:create_app", factory=True, host=settings.host, port=settings.port)
