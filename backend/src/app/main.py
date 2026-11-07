from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import main_router
from app.config import get_settings
from app.db.session import build_engine, create_session_factory


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Validate settings/.env at startup; the lifespan creates the
        # DB engine once at startup and closes it at shutdown.
        settings = get_settings()
        engine = build_engine(settings)
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(main_router)

    return app


app = create_app()
