import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, CameraSecret, Zone
from app.domain.camera import CameraSourceType, CameraStatus


def make_camera(code: str = "cam-01") -> Camera:
    return Camera(
        code=code,
        name="Corridor",
        source_type=CameraSourceType.RTSP,
        source_ref="rtsp://media-server/cam-01",
    )


def make_zone(camera_id: uuid.UUID) -> Zone:
    return Zone(
        camera_id=camera_id,
        name="Server room door",
        polygon=[[10, 10], [200, 10], [200, 150], [10, 150]],
        frame_width=1920,
        frame_height=1080,
    )


async def test_camera_gets_database_defaults(db_session: AsyncSession) -> None:
    camera = make_camera()
    db_session.add(camera)
    await db_session.flush()
    await db_session.refresh(camera)

    assert camera.status is CameraStatus.OFFLINE
    assert camera.enabled is True
    assert camera.access_settings == {}
    assert camera.last_seen_at is None


async def test_camera_code_must_be_unique(db_session: AsyncSession) -> None:
    db_session.add(make_camera(code="cam-01"))
    await db_session.flush()

    db_session.add(make_camera(code="cam-01"))
    with pytest.raises(IntegrityError, match="uq_cameras_code"):
        await db_session.flush()


async def test_unknown_source_type_is_rejected(db_session: AsyncSession) -> None:
    with pytest.raises(IntegrityError, match="ck_cameras_camerasourcetype"):
        await db_session.execute(
            text(
                "INSERT INTO cameras (code, name, source_type, source_ref) "
                "VALUES ('cam-x', 'X', 'http', 'x')"
            )
        )


async def test_zone_polygon_round_trips(db_session: AsyncSession) -> None:
    camera = make_camera()
    db_session.add(camera)
    await db_session.flush()

    zone = make_zone(camera.id)
    db_session.add(zone)
    await db_session.flush()

    stored = await db_session.scalar(select(Zone).where(Zone.id == zone.id))
    assert stored is not None
    assert stored.polygon == [[10, 10], [200, 10], [200, 150], [10, 150]]
    assert stored.version == 1
    assert stored.active is True


async def test_zone_needs_an_existing_camera(db_session: AsyncSession) -> None:
    db_session.add(make_zone(camera_id=uuid.uuid4()))
    with pytest.raises(IntegrityError, match="fk_zones_camera_id_cameras"):
        await db_session.flush()


async def test_camera_with_zones_cannot_be_deleted(db_session: AsyncSession) -> None:
    # Cameras are disabled, not deleted (BE-FR-60).
    camera = make_camera()
    db_session.add(camera)
    await db_session.flush()
    db_session.add(make_zone(camera.id))
    await db_session.flush()

    with pytest.raises(IntegrityError, match="fk_zones_camera_id_cameras"):
        await db_session.execute(
            text("DELETE FROM cameras WHERE id = :id"), {"id": camera.id}
        )


async def test_camera_secret_is_removed_with_its_camera(
    db_session: AsyncSession,
) -> None:
    camera = make_camera()
    db_session.add(camera)
    await db_session.flush()
    db_session.add(CameraSecret(camera_id=camera.id, secret_ref="env:CAM_01_PASSWORD"))
    await db_session.flush()

    await db_session.execute(
        text("DELETE FROM cameras WHERE id = :id"), {"id": camera.id}
    )

    remaining = await db_session.scalar(
        text("SELECT count(*) FROM camera_secrets WHERE camera_id = :id"),
        {"id": camera.id},
    )
    assert remaining == 0
