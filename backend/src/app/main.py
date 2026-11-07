from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import main_router
from app.config import get_settings


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Validate settings/.env at startup; the result is intentionally unused
        # because settings reach the code through the get_settings dependency.
        get_settings()
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(main_router)

    return app


app = create_app()
