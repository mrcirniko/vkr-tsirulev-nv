import os

from server import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    reload_enabled = os.getenv("ENV", "dev").strip().lower() != "production"
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=reload_enabled)
