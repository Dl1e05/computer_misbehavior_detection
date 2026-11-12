"""Rows that many model tests need as parents.

Fixtures, not importable helpers: with --import-mode=importlib, test modules
cannot import each other, but every test under tests/db sees these.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, Incident
from app.domain.camera import CameraSourceType
from app.domain.incident import IncidentStatus, IncidentType, Severity


@pytest.fixture
async def camera_id(db_session: AsyncSession) -> uuid.UUID:
    camera = Camera(
        code="cam-01",
        name="Corridor",
        source_type=CameraSourceType.RTSP,
        source_ref="rtsp://media-server/cam-01",
    )
    db_session.add(camera)
    await db_session.flush()
    return camera.id


@pytest.fixture
async def incident_id(db_session: AsyncSession, camera_id: uuid.UUID) -> uuid.UUID:
    incident = Incident(
        incident_type=IncidentType.FIGHT,
        severity=Severity.CRITICAL,
        status=IncidentStatus.SUSPECTED,
        camera_id=camera_id,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(incident)
    await db_session.flush()
    return incident.id
