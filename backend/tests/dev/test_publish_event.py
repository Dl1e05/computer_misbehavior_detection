import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.dev.publish_event import KINDS, build_event, main
from app.mqtt.events import EventLimits, MlEvent, parse_event

NOW = datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC)
LIMITS = EventLimits(received_at=NOW, max_future=timedelta(seconds=60))


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_kind_passes_validation(kind: str) -> None:
    # The simulator must follow the contract, or its tests prove nothing.
    event = build_event(
        kind, camera="cam-01", run_id="sim-1", zone=str(uuid.uuid4()), occurred_at=NOW
    )
    parsed = parse_event(json.dumps(event).encode(), LIMITS)

    assert isinstance(parsed, MlEvent), parsed
    assert (parsed.event_type.value, parsed.incident_type.value) == KINDS[kind]


def test_events_get_their_own_id() -> None:
    first, second = (
        build_event("fall", camera="cam-01", run_id="sim-1") for _ in range(2)
    )
    assert first["event_id"] != second["event_id"]


def test_fight_events_share_the_episode() -> None:
    suspicion = build_event("suspicion", camera="cam-01", run_id="r", episode="ep-7")
    confirmation = build_event(
        "confirmation", camera="cam-01", run_id="r", episode="ep-7"
    )
    assert suspicion["episode_id"] == confirmation["episode_id"] == "ep-7"


def test_zone_intrusion_needs_a_zone(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["zone_intrusion", "--dry-run"])
    assert "--zone" in capsys.readouterr().err


def test_dry_run_prints_without_publishing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No broker in this test: publishing would fail.
    main(["fall", "--count", "2", "--camera", "cam-02", "--dry-run"])

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["camera_id"] for line in lines} == {"cam-02"}


def test_file_is_published_byte_for_byte(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Invalid events must stay invalid, to test rejections.
    file = tmp_path / "broken.json"
    file.write_bytes(b'{"event_id": "not-a-uuid"')
    main(["--file", str(file), "--dry-run"])
    assert capsys.readouterr().out == '{"event_id": "not-a-uuid"\n'
