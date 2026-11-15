import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Clip
from app.domain.incident import ClipRecordState

START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
END = datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC)

# What an available clip needs (ck_clips_available_has_file).
FILE_FIELDS: dict[str, object] = {
    "location": "clips/2026/01/01/incident.mp4",
    "size_bytes": 5 * 1024**3,
    "checksum": "sha256:" + "0" * 64,
}


def make_clip(incident_id: uuid.UUID, **changes: object) -> Clip:
    clip = Clip(incident_id=incident_id, start_at=START, end_at=END)
    for name, value in changes.items():
        setattr(clip, name, value)
    return clip


async def test_clip_gets_database_defaults(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    clip = make_clip(incident_id)
    db_session.add(clip)
    await db_session.flush()
    await db_session.refresh(clip)

    assert clip.state is ClipRecordState.PENDING
    assert clip.location is None
    assert clip.error_code is None
    assert clip.version == 1


async def test_available_clip_with_file_is_accepted(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    # 5 GB: does not fit a 32-bit integer column, so this also proves BigInteger.
    clip = make_clip(incident_id, state=ClipRecordState.AVAILABLE, **FILE_FIELDS)
    db_session.add(clip)
    await db_session.flush()
    await db_session.refresh(clip)

    assert clip.size_bytes == 5 * 1024**3


async def test_one_clip_per_incident(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    db_session.add(make_clip(incident_id))
    await db_session.flush()

    db_session.add(make_clip(incident_id))
    with pytest.raises(IntegrityError, match="uq_clips_incident_id"):
        await db_session.flush()


@pytest.mark.parametrize(
    "bad",
    [
        *(pytest.param({name: None}, id=f"no {name}") for name in FILE_FIELDS),
        pytest.param({"location": ""}, id="empty location"),
        pytest.param({"checksum": ""}, id="empty checksum"),
        pytest.param({"size_bytes": 0}, id="0 bytes"),
    ],
)
async def test_available_clip_needs_its_file_fields(
    db_session: AsyncSession, incident_id: uuid.UUID, bad: dict[str, object]
) -> None:
    fields = {**FILE_FIELDS, **bad}
    db_session.add(make_clip(incident_id, state=ClipRecordState.AVAILABLE, **fields))
    with pytest.raises(IntegrityError, match="ck_clips_available_has_file"):
        await db_session.flush()


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"state": ClipRecordState.ERROR}, id="error without code"),
        pytest.param({"error_code": "DISK_FULL"}, id="pending with code"),
    ],
)
async def test_error_code_is_set_only_for_errors(
    db_session: AsyncSession, incident_id: uuid.UUID, changes: dict[str, object]
) -> None:
    db_session.add(make_clip(incident_id, **changes))
    with pytest.raises(IntegrityError, match="ck_clips_error_code_iff_error"):
        await db_session.flush()


async def test_error_with_code_is_accepted(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    db_session.add(
        make_clip(incident_id, state=ClipRecordState.ERROR, error_code="DISK_FULL")
    )
    await db_session.flush()


@pytest.mark.parametrize(
    ("changes", "constraint"),
    [
        pytest.param({"end_at": START}, "ck_clips_end_after_start", id="end == start"),
        pytest.param(
            {"size_bytes": -1}, "ck_clips_size_not_negative", id="negative size"
        ),
    ],
)
async def test_invalid_clip_is_rejected(
    db_session: AsyncSession,
    incident_id: uuid.UUID,
    changes: dict[str, object],
    constraint: str,
) -> None:
    db_session.add(make_clip(incident_id, **changes))
    with pytest.raises(IntegrityError, match=constraint):
        await db_session.flush()


async def test_incident_with_clip_cannot_be_deleted(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    # The archive file must be removed before the row (no CASCADE).
    db_session.add(make_clip(incident_id))
    await db_session.flush()

    with pytest.raises(IntegrityError, match="fk_clips_incident_id_incidents"):
        await db_session.execute(
            text("DELETE FROM incidents WHERE id = :id"), {"id": incident_id}
        )
