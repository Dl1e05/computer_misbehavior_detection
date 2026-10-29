from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/live")
async def health_live() -> HealthResponse:
    return HealthResponse()
