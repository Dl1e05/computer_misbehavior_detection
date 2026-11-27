"""Development simulator: publishes ML events to the broker, like ML would.

Examples (from backend/, with the dev broker running):

    uv run python -m app.dev.publish_event fall
    uv run python -m app.dev.publish_event fire --camera cam-02
    uv run python -m app.dev.publish_event zone_intrusion --zone <zone uuid>
    uv run python -m app.dev.publish_event suspicion --episode ep-7
    uv run python -m app.dev.publish_event confirmation --episode ep-7
    uv run python -m app.dev.publish_event fall --count 3 --interval 5
    uv run python -m app.dev.publish_event --file my_event.json
    uv run python -m app.dev.publish_event fall --dry-run

Each generated event gets a new event_id and occurred_at = now. A --file is
published byte for byte, so invalid events can be tested too.
"""

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiomqtt

from app.config import get_settings

# kind -> (event_type, incident_type)
KINDS = {
    "fall": ("detection", "fall"),
    "fire": ("detection", "fire"),
    "zone_intrusion": ("detection", "zone_intrusion"),
    "suspicion": ("suspicion", "fight"),
    "confirmation": ("confirmation", "fight"),
}


DEFAULT_CAMERA = "cam-01"


def event_topic(camera: str) -> str:
    return f"videoanalytics/v1/cameras/{camera}/events"


def camera_for(camera: str | None, file: Path | None) -> str:
    """--camera if given, else the file's camera_id, else cam-01.

    The topic must name the event's camera, or BE rejects the event.
    """
    if camera is not None:
        return camera
    if file is not None:
        try:
            loaded = json.loads(file.read_bytes())
        except ValueError:  # a broken file is published anyway
            loaded = None
        if isinstance(loaded, dict):
            from_file = loaded.get("camera_id")
            if isinstance(from_file, str) and from_file:
                return from_file
    return DEFAULT_CAMERA


def build_event(
    kind: str,
    *,
    camera: str,
    run_id: str,
    zone: str | None = None,
    episode: str = "episode-1",
    occurred_at: datetime | None = None,
) -> dict[str, Any]:
    """A valid event of the given kind, shaped like ML's own messages."""
    event_type, incident_type = KINDS[kind]
    when = occurred_at or datetime.now(UTC)
    event: dict[str, Any] = {
        "schema_version": "1.0",
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "camera_id": camera,
        "occurred_at": when.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "incident_type": incident_type,
        "run_id": run_id,
        "model_version": "simulator-1.0",
        "confidence": 0.9,
        "participants": [
            {"track_id": 1, "confidence": 0.9, "bbox": [0.3, 0.2, 0.5, 0.9]}
        ],
    }
    if incident_type == "fire":
        event["participants"] = []
        event["bbox"] = [0.4, 0.4, 0.6, 0.7]
    if incident_type == "fight":
        event["episode_id"] = episode
    if zone is not None:
        event["zone_id"] = zone
        event["zone_version"] = 1
    return event


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish ML events to the MQTT broker (development only)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("kind", nargs="?", choices=sorted(KINDS))
    source.add_argument("--file", type=Path, help="publish this JSON file as is")
    parser.add_argument(
        "--camera", help="camera code (default: the file's camera_id, or cam-01)"
    )
    parser.add_argument("--zone", help="zone UUID (required for zone_intrusion)")
    parser.add_argument("--episode", default="episode-1", help="fight episode_id")
    parser.add_argument("--run-id", default=f"sim-{datetime.now(UTC):%Y%m%dT%H%M%S}")
    parser.add_argument("--count", type=int, default=1, help="how many events")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between")
    parser.add_argument("--dry-run", action="store_true", help="print, do not publish")
    args = parser.parse_args(argv)
    if args.kind == "zone_intrusion" and args.zone is None:
        parser.error("zone_intrusion needs --zone <zone uuid>")
    if args.count < 1:
        parser.error("--count must be at least 1")
    return args


def _message(args: argparse.Namespace) -> bytes:
    # A new event each time, so every message has its own event_id and
    # occurred_at is the time it is sent.
    if args.file is not None:
        return bytes(args.file.read_bytes())
    event = build_event(
        args.kind,
        camera=camera_for(args.camera, None),
        run_id=args.run_id,
        zone=args.zone,
        episode=args.episode,
    )
    return json.dumps(event).encode()


async def _publish(args: argparse.Namespace) -> None:
    topic = event_topic(camera_for(args.camera, args.file))
    settings = get_settings()
    async with aiomqtt.Client(
        settings.mqtt_host,
        settings.mqtt_port,
        username=settings.mqtt_username,
        password=settings.mqtt_password.get_secret_value(),
        # Never the worker's id: the same id would disconnect the worker.
        identifier=f"simulator-{uuid.uuid4().hex[:8]}",
    ) as client:
        for number in range(args.count):
            if number:
                await asyncio.sleep(args.interval)
            payload = _message(args)
            # QoS 1, like ML: returns once the broker has the message.
            await client.publish(topic, payload, qos=1)
            print(f"published to {topic}: {payload.decode(errors='replace')}")


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    if args.dry_run:
        for _ in range(args.count):
            print(_message(args).decode(errors="replace"))
        return
    try:
        asyncio.run(_publish(args))
    except aiomqtt.MqttError as error:
        sys.exit(f"could not publish: {error}")


if __name__ == "__main__":
    main()
