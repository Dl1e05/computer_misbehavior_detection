"""Timer worker: fires due timers every second (SRS §6.5).

Run: python -m app.workers.timers

One instance is enough. More are safe (SKIP LOCKED: no timer fires twice) but
not faster: each reads the same due timers.
"""

import asyncio
import logging
import signal
import uuid
from datetime import UTC, datetime

from app.config import Settings, get_settings
from app.db.errors import describe, is_database_unavailable
from app.db.session import (
    build_engine,
    create_session_factory,
    wait_for_database,
)
from app.services.timers import fire_due_timers

logger = logging.getLogger(__name__)


TICK = 1.0  # seconds (SRS §6.5)
MAX_DELAY = 30.0


async def run(settings: Settings) -> None:
    engine = build_engine(settings)
    try:
        await wait_for_database(engine, first_delay=TICK, max_delay=MAX_DELAY)
        sessions = create_session_factory(engine)
        # Timers that failed in this process; tried again after a restart.
        broken: set[uuid.UUID] = set()
        delay = TICK
        while True:
            try:
                fired = await fire_due_timers(
                    sessions, datetime.now(UTC), broken=broken
                )
            except Exception as error:
                # Errors of single timers are handled inside; anything else
                # is a bug and stops the worker loudly.
                if not is_database_unavailable(error):
                    raise
                logger.warning(
                    "database not reachable (%s), retry in %.0fs",
                    describe(error),
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)
                continue
            if fired:
                logger.info("fired %d timers", fired)
            delay = TICK
            await asyncio.sleep(TICK)
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
        logger.info("timer worker stopped")


if __name__ == "__main__":
    main()
