import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Notification
from app.domain.incident import Role
from app.domain.notification import INCIDENT_NOTIFICATION_KINDS, NotificationKind

SYSTEM_KINDS = sorted(set(NotificationKind) - INCIDENT_NOTIFICATION_KINDS)


def make_notification(**changes: object) -> Notification:
    notification = Notification(
        kind=NotificationKind.CAMERA_STATUS,
        target_role=Role.OPERATOR,
        payload_ref="notifications/payload-1",
    )
    for name, value in changes.items():
        setattr(notification, name, value)
    return notification


async def test_notification_gets_database_defaults(db_session: AsyncSession) -> None:
    notification = make_notification()
    db_session.add(notification)
    await db_session.flush()
    await db_session.refresh(notification)

    assert notification.delivered_at is None
    assert notification.created_at.tzinfo is not None
    assert notification.version == 1


@pytest.mark.parametrize("kind", sorted(INCIDENT_NOTIFICATION_KINDS))
async def test_incident_kinds_are_accepted_with_incident(
    db_session: AsyncSession, incident_id: uuid.UUID, kind: NotificationKind
) -> None:
    db_session.add(make_notification(kind=kind, incident_id=incident_id))
    await db_session.flush()


@pytest.mark.parametrize("kind", sorted(INCIDENT_NOTIFICATION_KINDS))
async def test_incident_kinds_need_an_incident(
    db_session: AsyncSession, kind: NotificationKind
) -> None:
    db_session.add(make_notification(kind=kind))
    with pytest.raises(
        IntegrityError, match="ck_notifications_incident_id_iff_incident_kind"
    ):
        await db_session.flush()


@pytest.mark.parametrize("kind", SYSTEM_KINDS)
async def test_system_kinds_are_accepted_without_incident(
    db_session: AsyncSession, kind: NotificationKind
) -> None:
    db_session.add(make_notification(kind=kind))
    await db_session.flush()


@pytest.mark.parametrize("kind", SYSTEM_KINDS)
async def test_system_kinds_must_not_have_an_incident(
    db_session: AsyncSession, incident_id: uuid.UUID, kind: NotificationKind
) -> None:
    db_session.add(make_notification(kind=kind, incident_id=incident_id))
    with pytest.raises(
        IntegrityError, match="ck_notifications_incident_id_iff_incident_kind"
    ):
        await db_session.flush()


async def test_notifications_are_removed_with_their_incident(
    db_session: AsyncSession, incident_id: uuid.UUID
) -> None:
    db_session.add(
        make_notification(kind=NotificationKind.ALARM, incident_id=incident_id)
    )
    await db_session.flush()

    await db_session.execute(
        text("DELETE FROM incidents WHERE id = :id"), {"id": incident_id}
    )

    remaining = await db_session.scalar(
        text("SELECT count(*) FROM notifications WHERE incident_id = :id"),
        {"id": incident_id},
    )
    assert remaining == 0
