from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from memcraft import __version__
from memcraft.db import database_ok

FRONTEND = Path(__file__).resolve().parents[3] / "frontend"

app = FastAPI(title="MemcraftAI", version=__version__, docs_url=None, redoc_url=None)


@app.get("/health")
def health(response: Response) -> dict[str, str]:
    """200 when the app and the database both answer, 503 when the database does not."""
    ok = database_ok()
    if not ok:
        response.status_code = 503
    return {
        "status": "ok" if ok else "degraded",
        "database": "ok" if ok else "unreachable",
        "version": __version__,
    }


# Last, so the API routes above win. Plain HTML, CSS and JS: no Node build step.
if FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
