from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from memcraft import __version__
from memcraft.config import Settings, get_settings
from memcraft.db import database_ok
from memcraft.web import admin, api, faces, upload

FRONTEND = Path(__file__).resolve().parents[3] / "frontend"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob:; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
}


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(
        title="MemcraftAI", version=__version__, docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=settings.allowed_host_list, www_redirect=False
    )

    @app.get("/health")
    def health(response: Response, ok: Annotated[bool, Depends(database_ok)]) -> dict[str, str]:
        """200 when the app and the database both answer, 503 when the database does not."""
        if not ok:
            response.status_code = 503
        return {
            "status": "ok" if ok else "degraded",
            "database": "ok" if ok else "unreachable",
            "version": __version__,
        }

    app.include_router(api.router)
    app.include_router(upload.router)
    app.include_router(admin.router)
    app.include_router(faces.router)

    # Last, so the routes above win. Plain HTML, CSS and JS: no Node build step.
    if FRONTEND.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    return app


app = create_app(get_settings())
