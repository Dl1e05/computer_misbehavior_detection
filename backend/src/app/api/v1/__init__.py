from fastapi import APIRouter

from app.api.v1.health import router as health_router

main_router = APIRouter(prefix="/api/v1")
main_router.include_router(health_router)

__all__ = ["main_router"]
