"""Rows that many model tests need as parents.

Fixtures, not importable helpers: with --import-mode=importlib, test modules
cannot import each other, but every test under tests/db sees these.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, Incident, Zone
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
async def zone_id(db_session: AsyncSession, camera_id: uuid.UUID) -> uuid.UUID:
    zone = Zone(
        camera_id=camera_id,
        name="Server room door",
        polygon=[[10, 10], [200, 10], [200, 150], [10, 150]],
        frame_width=1920,
        frame_height=1080,
    )
    db_session.add(zone)
    await db_session.flush()
    return zone.id


@pytest.fixture
async def incident_id(db_session: AsyncSession, camera_id: uuid.UUID) -> uuid.UUID:
    incident = Incident(
        incident_type=IncidentType.FIGHT,
        severity=Severity.CRITICAL,
        status=IncidentStatus.SUSPECTED,
        camera_id=camera_id,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        created_received_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    db_session.add(incident)
    await db_session.flush()
    return incident.id
