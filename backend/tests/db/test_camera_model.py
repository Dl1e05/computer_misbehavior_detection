import uuid
from typing import Any

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
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
        polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]],
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

    # Forget the in-memory object, so the next select really reads the row.
    # Keep the id first: reading an attribute of an expired object would
    # trigger a lazy load, which is not allowed under AsyncSession.
    zone_id = zone.id
    db_session.expire_all()
    stored = await db_session.scalar(select(Zone).where(Zone.id == zone_id))
    assert stored is not None
    assert stored.polygon == [
        [0.1, 0.1],
        [0.9, 0.1],
        [0.9, 0.8],
        [0.1, 0.8],
    ]
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


async def test_disabled_is_not_a_camera_status(db_session: AsyncSession) -> None:
    # "Switched off" is Camera.enabled; status is only the connection state.
    # The column is VARCHAR(longest value), so 'disabled' is refused as too
    # long even before the CHECK; either refusal is fine.
    with pytest.raises(DBAPIError):
        await db_session.execute(
            text(
                "INSERT INTO cameras (code, name, source_type, source_ref, status) "
                "VALUES ('cam-x', 'X', 'rtsp', 'x', 'disabled')"
            )
        )


async def test_in_place_json_changes_are_saved(db_session: AsyncSession) -> None:
    camera = make_camera()
    db_session.add(camera)
    await db_session.flush()
    zone = make_zone(camera.id)
    db_session.add(zone)
    await db_session.flush()

    camera.access_settings["allowed_roles"] = ["OPERATOR"]
    zone.polygon.append([0.5, 0.5])
    await db_session.flush()

    camera_id, zone_id = camera.id, zone.id
    db_session.expire_all()
    stored_camera = await db_session.scalar(
        select(Camera).where(Camera.id == camera_id)
    )
    stored_zone = await db_session.scalar(select(Zone).where(Zone.id == zone_id))
    assert stored_camera is not None
    assert stored_zone is not None
    assert stored_camera.access_settings == {"allowed_roles": ["OPERATOR"]}
    assert len(stored_zone.polygon) == 5


@pytest.mark.parametrize(
    ("polygon", "constraint"),
    [
        pytest.param([[0, 0], [1, 1]], "ck_zones_polygon_min_3_points", id="2 points"),
        pytest.param([], "ck_zones_polygon_min_3_points", id="empty"),
        *[
            pytest.param(
                [[0, 0], [1, 0], point], "ck_zones_polygon_normalized", id=str(point)
            )
            for point in (
                [1, 1.5],
                [-0.1, 1],
                [960, 540],
                [1, 1, 1],
                [1],
                [],
                ["1", 1],
                [None, 1],
                [True, 1],
                [[0, 1], 1],
                0.5,
                None,
                {"x": 1, "y": 1},
            )
        ],
        # Either CHECK may report these first.
        pytest.param({"points": []}, "ck_zones_polygon_", id="object"),
        pytest.param("square", "ck_zones_polygon_", id="text"),
    ],
)
async def test_invalid_zone_geometry_is_rejected(
    db_session: AsyncSession, polygon: Any, constraint: str
) -> None:
    camera = make_camera()
    db_session.add(camera)
    await db_session.flush()

    # Not through the model: MutableList itself refuses what is not a list.
    with pytest.raises(IntegrityError, match=constraint):
        await db_session.execute(
            insert(Zone).values(camera_id=camera.id, name="Door", polygon=polygon)
        )


async def test_zone_may_touch_the_frame_edges(db_session: AsyncSession) -> None:
    camera = make_camera()
    db_session.add(camera)
    await db_session.flush()

    zone = make_zone(camera.id)
    zone.polygon = [[0, 0], [1, 0], [1.0, 1.0], [0.0, 1]]
    db_session.add(zone)
    await db_session.flush()
