"""One MQTT event message in, one acknowledgement decision out (SRS §7.1, §7.5).

The broker sends an unacknowledged message again only after a reconnect. So
ACK means "saved, acknowledge it", and RETRY means "nothing was saved: do not
acknowledge, reconnect, and get it again".
"""

import hashlib
import logging
from collections import Counter
from datetime import datetime, timedelta
from enum import StrEnum

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import DeadLetter
from app.domain.event import FailureCode
from app.mqtt.events import EventLimits
from app.services.incidents import IncidentTimings
from app.services.ingest import ingest_event

logger = logging.getLogger(__name__)

# A message that fails this often for a reason other than a lost connection
# is given up, so it cannot block every message behind it forever.
MAX_ATTEMPTS = 3


class Ack(StrEnum):
    ACK = "ack"
    RETRY = "retry"


class EventMessageHandler:
    """Processes event messages one at a time; not safe for parallel use."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._max_future = timedelta(seconds=settings.event_max_future_seconds)
        self._timings = IncidentTimings.from_settings(settings)
        # Failed attempts per payload hash, for this process only. An entry
        # is removed once its message is saved or given up.
        self._failures: Counter[str] = Counter()

    async def handle(self, topic: str, payload: bytes, received_at: datetime) -> Ack:
        """Saves the outcome of one message and commits; never raises."""
        payload_hash = hashlib.sha256(payload).hexdigest()
        limits = EventLimits(received_at=received_at, max_future=self._max_future)
        try:
            async with self._session_factory() as session:
                outcome = await ingest_event(
                    session, payload, topic=topic, limits=limits, timings=self._timings
                )
                await session.commit()
        except Exception as error:
            return await self._failed(error, topic, payload_hash, received_at)

        self._failures.pop(payload_hash, None)
        logger.info(
            "event %s: topic=%s hash=%s", outcome.status.value, topic, payload_hash
        )
        return Ack.ACK

    async def _failed(
        self, error: Exception, topic: str, payload_hash: str, received_at: datetime
    ) -> Ack:
        if _is_connection_error(error):
            # Not counted: a valid alarm must not become a dead letter only
            # because the database was restarting.
            logger.warning(
                "event not saved, database unavailable (%s): topic=%s hash=%s",
                _describe(error),
                topic,
                payload_hash,
            )
            return Ack.RETRY

        self._failures[payload_hash] += 1
        attempt = self._failures[payload_hash]
        logger.error(
            "event processing failed, attempt %d of %d (%s): topic=%s hash=%s",
            attempt,
            MAX_ATTEMPTS,
            _describe(error),
            topic,
            payload_hash,
            # Our own errors get a traceback; database errors would print the
            # SQL parameters, which hold the event's values.
            exc_info=not isinstance(error, DBAPIError),
        )
        if attempt < MAX_ATTEMPTS:
            # A race on a unique key passes on the next try.
            return Ack.RETRY

        try:
            async with self._session_factory() as session:
                # Hash only: the payload is what could not be saved.
                session.add(
                    DeadLetter(
                        received_at=received_at,
                        topic=topic,
                        failure_code=FailureCode.PROCESSING_ERROR.value,
                        payload_hash=payload_hash,
                        raw_payload=None,
                    )
                )
                await session.commit()
        except Exception as dead_letter_error:
            # Kept unacknowledged; the count stays, so the next try gives up
            # again at once.
            logger.error(
                "dead letter not saved (%s): topic=%s hash=%s",
                _describe(dead_letter_error),
                topic,
                payload_hash,
            )
            return Ack.RETRY

        del self._failures[payload_hash]
        logger.error(
            "event given up after %d attempts: topic=%s hash=%s",
            MAX_ATTEMPTS,
            topic,
            payload_hash,
        )
        return Ack.ACK


def _is_connection_error(error: Exception) -> bool:
    # A refused connection arrives as a plain OSError, not wrapped by
    # SQLAlchemy; a connection lost during a query is a DBAPIError that
    # invalidated the connection.
    if isinstance(error, DBAPIError) and error.connection_invalidated:
        return True
    return isinstance(error, (OperationalError, InterfaceError, OSError))


def _describe(error: Exception) -> str:
    # Class names only: the message of a database error holds SQL parameters.
    if isinstance(error, DBAPIError):
        return f"{type(error).__name__}: {type(error.orig).__name__}"
    return type(error).__name__
