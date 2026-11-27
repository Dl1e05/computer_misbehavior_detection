"""Event worker: receives ML events over MQTT and saves them (SRS §7.1).

Run one instance: python -m app.workers.events

A separate process, not part of the API: several API workers would share one
MQTT client id and keep disconnecting each other, and restarting the API must
not stop event intake.
"""

import asyncio
import logging
import signal
from collections.abc import Callable
from datetime import UTC, datetime

import aiomqtt
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings, get_settings
from app.db.session import (
    build_engine,
    check_app_role_is_limited,
    create_session_factory,
)
from app.mqtt.consumer import (
    Ack,
    EventMessageHandler,
    describe,
    is_database_unavailable,
)

logger = logging.getLogger(__name__)

EVENT_TOPICS = "videoanalytics/v1/cameras/+/events"
# Pause before reconnecting: doubles after each failed connection.
FIRST_DELAY = 1.0
MAX_DELAY = 30.0


class _NotSaved(Exception):
    """A message was not saved; reconnecting makes the broker send it again."""


def new_client(settings: Settings) -> aiomqtt.Client:
    client = aiomqtt.Client(
        settings.mqtt_host,
        settings.mqtt_port,
        username=settings.mqtt_username,
        password=settings.mqtt_password.get_secret_value(),
        identifier=settings.mqtt_client_id,
        # Persistent session (BE-FR-01): the broker keeps our messages while
        # we are offline and sends unacknowledged ones again on reconnect.
        clean_session=False,
    )
    # paho acknowledges a QoS 1 message as soon as it arrives, before we have
    # saved it; a crash would then lose the alarm. aiomqtt 2.5 has no option
    # for manual acknowledgement, so it is set on the paho client inside.
    # Before connecting: the broker sends stored messages right away.
    client._client.manual_ack_set(True)
    return client


def acknowledge(client: aiomqtt.Client, message: aiomqtt.Message) -> None:
    client._client.ack(message.mid, message.qos)


async def consume(
    client: aiomqtt.Client,
    handler: EventMessageHandler,
    on_saved: Callable[[], None],
) -> None:
    """Saves and acknowledges messages one at a time, in arrival order.

    Raises _NotSaved when a message must come again; aiomqtt.MqttError when
    the connection is lost.
    """
    await client.subscribe(EVENT_TOPICS, qos=1)
    async for message in client.messages:
        received_at = datetime.now(UTC)
        ack = await handler.handle(str(message.topic), message.payload, received_at)
        if ack is Ack.RETRY:
            raise _NotSaved
        acknowledge(client, message)
        on_saved()


async def wait_for_database(engine: AsyncEngine) -> None:
    """Checks the database login, waiting while the database is unreachable.

    The worker may start before Postgres is ready (compose, reboot). An
    unsafe login still stops it at once.
    """
    delay = FIRST_DELAY
    while True:
        try:
            await check_app_role_is_limited(engine)
        except Exception as error:
            if not is_database_unavailable(error):
                raise
            logger.warning(
                "database not reachable (%s), retry in %.0fs", describe(error), delay
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_DELAY)
        else:
            return


async def run(settings: Settings) -> None:
    """Consumes until cancelled, reconnecting after every failure.

    Cancelling is safe at any point: an unsaved message is rolled back, and
    a saved but unacknowledged one comes again and is a duplicate.
    """
    engine = build_engine(settings)
    try:
        await wait_for_database(engine)
        handler = EventMessageHandler(create_session_factory(engine), settings)
        delay = FIRST_DELAY

        def saved() -> None:
            # Reset only after progress: connecting alone proves nothing, and
            # a message that keeps failing must not be retried every second.
            nonlocal delay
            delay = FIRST_DELAY

        while True:
            try:
                async with new_client(settings) as client:
                    logger.info("connected to the MQTT broker")
                    await consume(client, handler, saved)
            except aiomqtt.MqttError as error:
                logger.warning(
                    "MQTT connection lost (%s), retry in %.0fs", error, delay
                )
            except _NotSaved:
                logger.warning("message not saved, reconnecting in %.0fs", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_DELAY)
    finally:
        await engine.dispose()


async def _main() -> None:
    task = asyncio.current_task()
    if task is not None:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    await run(get_settings())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("event worker stopped")


if __name__ == "__main__":
    main()
