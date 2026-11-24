import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from app.domain.event import EventType, FailureCode
from app.domain.incident import IncidentType
from app.mqtt.events import EventLimits, MlEvent

LIMITS = EventLimits(
    received_at=datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC),
    max_future=timedelta(seconds=60),
)


def srs_sample() -> dict[str, Any]:
    """The example event from SRS §7, in the form agreed with the ML team."""
    return {
        "schema_version": "1.0",
        "event_id": "5f0c2a7e-3b9d-4c1a-9e8f-2d7b6a1c0e42",
        "event_type": "confirmation",
        "camera_id": "cam-02",
        "occurred_at": "2026-10-25T09:14:03.217Z",
        "incident_type": "fight",
        "run_id": "ml-20261025-091000",
        "model_version": "fight-temporal-1.3.0/yolo26-det-2.0.1/bytetrack-1.0",
        "participants": [
            {
                "track_id": 17,
                "confidence": 0.91,
                "bbox": [0.25, 0.125, 0.375, 0.5],
            },
            {
                "track_id": 23,
                "confidence": 0.88,
                "bbox": [0.5, 0.125, 0.75, 0.625],
            },
        ],
    }


def parse(payload: dict[str, Any], limits: EventLimits = LIMITS) -> MlEvent:
    # From raw JSON text, the way it arrives over MQTT.
    return MlEvent.model_validate_json(json.dumps(payload), context=limits)


def failure_code(payload: dict[str, Any]) -> str:
    """The error type of the first error: the SRS code for rule violations."""
    with pytest.raises(ValidationError) as error:
        parse(payload)
    return error.value.errors()[0]["type"]


def detection(incident_type: str = "fall", **changes: Any) -> dict[str, Any]:
    return srs_sample() | {
        "event_type": "detection",
        "incident_type": incident_type,
        **changes,
    }


def test_srs_sample_parses() -> None:
    event = parse(srs_sample())

    assert event.event_type is EventType.CONFIRMATION
    assert event.incident_type is IncidentType.FIGHT
    assert event.occurred_at.tzinfo is not None
    assert event.run_id == "ml-20261025-091000"
    assert [p.track_id for p in event.participants] == [17, 23]
    assert event.participants[0].bbox == [0.25, 0.125, 0.375, 0.5]
    assert event.zone_id is None
    assert event.zone_version is None
    assert event.confidence is None
    assert event.bbox is None
    assert event.episode_id is None


def test_optional_fields_have_defaults() -> None:
    # A suspicion may come without participants (V12 needs them only on
    # incident-creating events).
    payload = srs_sample() | {"event_type": "suspicion"}
    del payload["participants"]
    event = parse(payload)

    assert event.participants == []
    assert event.sent_at is None


def test_unknown_fields_are_ignored() -> None:
    # A field added in a newer minor schema version must not reject the event.
    payload = srs_sample() | {"line_id": "line-7"}
    payload["participants"][0]["new_field"] = 1

    event = parse(payload)
    assert not hasattr(event, "line_id")


def test_episode_id_is_kept() -> None:
    event = parse(srs_sample() | {"episode_id": "ep-42"})
    assert event.episode_id == "ep-42"


@pytest.mark.parametrize("track_id", [0, 2**63 - 1])
def test_track_id_limits_are_accepted(track_id: int) -> None:
    payload = srs_sample()
    payload["participants"][0]["track_id"] = track_id
    assert parse(payload).participants[0].track_id == track_id


def test_whole_numbers_are_accepted_in_bbox() -> None:
    # The whole frame: 0 and 1 are allowed.
    payload = srs_sample() | {"bbox": [0, 0, 1, 1]}
    payload["participants"][0]["bbox"] = [0, 0, 1, 1]
    event = parse(payload)
    assert event.bbox == [0.0, 0.0, 1.0, 1.0]
    assert event.participants[0].bbox == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "bbox",
    [
        pytest.param([1.0, 0.1, 1.0, 0.9], id="no width"),
        pytest.param([0.1, 0.5, 0.9, 0.5], id="no height"),
    ],
)
def test_bbox_without_area_is_accepted(bbox: list[float]) -> None:
    # A person clipped at the frame edge; the alarm is real.
    payload = detection("fall")
    payload["participants"][0] = payload["participants"][0] | {"bbox": bbox}
    assert parse(payload).participants[0].bbox == bbox
    assert parse(detection("fire", bbox=bbox)).bbox == bbox


INVALID_BOXES = [
    pytest.param([0.1, 0.1, 0.9], id="3 numbers"),
    pytest.param([0.1, 0.1, 0.9, 0.9, 0.9], id="5 numbers"),
    pytest.param([412.0, 188.5, 498.0, 402.0], id="pixels"),
    pytest.param([0.1, 0.1, 1.1, 0.9], id="above 1"),
    pytest.param([-0.1, 0.1, 0.9, 0.9], id="below 0"),
    pytest.param([0.9, 0.1, 0.1, 0.9], id="x reversed"),
    pytest.param([0.1, 0.9, 0.9, 0.1], id="y reversed"),
    pytest.param([0.1, 0.1, math.nan, 0.9], id="NaN"),
    pytest.param([0.1, 0.1, math.inf, 0.9], id="Infinity"),
    pytest.param(["0.1", 0.1, 0.9, 0.9], id="number as string"),
]


@pytest.mark.parametrize("bbox", INVALID_BOXES)
def test_invalid_participant_bbox_is_rejected(bbox: list[Any]) -> None:
    payload = srs_sample()
    payload["participants"][0]["bbox"] = bbox
    with pytest.raises(ValidationError) as error:
        parse(payload)
    assert error.value.errors()[0]["loc"][:3] == ("participants", 0, "bbox")


@pytest.mark.parametrize("bbox", INVALID_BOXES)
def test_invalid_event_bbox_is_rejected(bbox: list[Any]) -> None:
    with pytest.raises(ValidationError) as error:
        parse(detection("fire", bbox=bbox))
    assert error.value.errors()[0]["loc"][0] == "bbox"


def test_event_details_are_kept() -> None:
    event = parse(
        detection(
            "zone_intrusion", zone_id="zone-1", zone_version=2**31 - 1, confidence=0.87
        )
    )
    assert event.zone_version == 2**31 - 1
    assert event.confidence == 0.87


# Less than ML promises is tolerated where refusing would lose a real alarm.
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(detection("fire", participants=[]), id="fire without bbox"),
        pytest.param(detection("fire", bbox=[0, 0, 1, 1]), id="fire with people"),
        pytest.param(detection("fall", bbox=[0, 0, 1, 1]), id="bbox on a fall"),
        pytest.param(srs_sample() | {"episode_id": None}, id="fight without episode"),
        pytest.param(
            detection("zone_intrusion", zone_id="zone-1", zone_version=None),
            id="zone without version",
        ),
        pytest.param(detection("fall", confidence=None), id="null confidence"),
    ],
)
def test_incomplete_event_is_accepted(payload: dict[str, Any]) -> None:
    assert parse(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("event_id", "not-a-uuid", id="bad event_id"),
        pytest.param("event_type", "telepathy", id="unknown event_type"),
        pytest.param("incident_type", "theft", id="unknown incident_type"),
        pytest.param("occurred_at", "2026-10-25T09:14:03", id="time without zone"),
        pytest.param("occurred_at", "yesterday", id="not a time"),
        pytest.param("model_version", "x" * 129, id="model_version too long"),
        pytest.param("camera_id", "", id="empty camera_id"),
        pytest.param("schema_version", 1.0, id="schema_version not a string"),
        pytest.param("participants", "17,23", id="participants not a list"),
        pytest.param("run_id", "", id="empty run_id"),
        pytest.param("run_id", "r" * 129, id="run_id too long"),
        pytest.param("run_id", 7, id="run_id as number"),
        pytest.param("zone_version", 2**31, id="zone_version too large"),
        pytest.param("zone_version", -1, id="negative zone_version"),
        pytest.param("zone_version", "1", id="zone_version as string"),
        pytest.param("confidence", 1.5, id="event confidence > 1"),
    ],
)
def test_invalid_event_field_is_rejected(field: str, value: object) -> None:
    payload = srs_sample() | {field: value}
    with pytest.raises(ValidationError) as error:
        parse(payload)
    assert error.value.errors()[0]["loc"][0] == field


@pytest.mark.parametrize(
    "required", ["schema_version", "event_id", "event_type", "camera_id", "run_id"]
)
def test_missing_required_field_is_rejected(required: str) -> None:
    payload = srs_sample()
    del payload[required]
    with pytest.raises(ValidationError, match="Field required"):
        parse(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("confidence", 1.5, id="confidence > 1"),
        pytest.param("confidence", "0.9", id="confidence as string"),
        pytest.param("confidence", math.nan, id="confidence NaN"),
        pytest.param("track_id", "17", id="track_id as string"),
        pytest.param("track_id", True, id="track_id as bool"),
        pytest.param("track_id", 1.0, id="track_id as float"),
        pytest.param("track_id", -1, id="negative track_id"),
        pytest.param("track_id", 2**63, id="track_id too large"),
        pytest.param("track_id", None, id="null track_id"),
    ],
)
def test_invalid_participant_field_is_rejected(field: str, value: object) -> None:
    payload = srs_sample()
    payload["participants"][0][field] = value
    with pytest.raises(ValidationError) as error:
        parse(payload)
    assert error.value.errors()[0]["loc"][:3] == ("participants", 0, field)


def test_event_is_immutable() -> None:
    # Once validated, nothing downstream can change what ML sent.
    event = parse(srs_sample())
    with pytest.raises(ValidationError):
        event.camera_id = "cam-99"


def test_limits_are_required() -> None:
    # Without them V9 would be skipped silently, so it is an error.
    with pytest.raises(TypeError, match="EventLimits"):
        MlEvent.model_validate_json(json.dumps(srs_sample()))


# V3
@pytest.mark.parametrize("version", ["1.0", "1.7", "1.12"])
def test_any_minor_version_of_schema_1_is_accepted(version: str) -> None:
    assert parse(srs_sample() | {"schema_version": version})


@pytest.mark.parametrize("version", ["2.0", "0.9", "1", "v1.0", "1.0.0"])
def test_other_schema_versions_are_rejected(version: str) -> None:
    payload = srs_sample() | {"schema_version": version}
    assert failure_code(payload) == FailureCode.UNSUPPORTED_SCHEMA


# V4
def test_event_id_must_be_uuid_v4() -> None:
    payload = srs_sample() | {"event_id": "5f0c2a7e-3b9d-1c1a-9e8f-2d7b6a1c0e42"}
    assert failure_code(payload) == FailureCode.INVALID_EVENT_ID


# V7
@pytest.mark.parametrize(
    ("event_type", "incident_type"),
    [
        ("suspicion", "fight"),
        ("confirmation", "fight"),
        ("detection", "fall"),
        ("detection", "fire"),
    ],
)
def test_allowed_stage_type_pairs(event_type: str, incident_type: str) -> None:
    payload = srs_sample() | {"event_type": event_type, "incident_type": incident_type}
    assert parse(payload).event_type == event_type


@pytest.mark.parametrize(
    ("event_type", "incident_type"),
    [
        ("detection", "fight"),
        ("suspicion", "fall"),
        ("confirmation", "fire"),
        ("suspicion", "zone_intrusion"),
    ],
)
def test_forbidden_stage_type_pairs(event_type: str, incident_type: str) -> None:
    payload = srs_sample() | {
        "event_type": event_type,
        "incident_type": incident_type,
        "zone_id": "zone-1",
    }
    assert failure_code(payload) == FailureCode.INVALID_STAGE_TYPE_PAIR


# V9
def test_time_must_be_utc() -> None:
    payload = srs_sample() | {"occurred_at": "2026-10-25T14:14:03.217+05:00"}
    assert failure_code(payload) == FailureCode.INVALID_TIMESTAMP


def test_small_future_skew_is_accepted() -> None:
    # Within the tolerance it is only a clock-skew warning (V10).
    payload = srs_sample() | {"occurred_at": "2026-10-25T09:15:00Z"}
    assert parse(payload)


def test_far_future_time_is_rejected() -> None:
    payload = srs_sample() | {"occurred_at": "2026-10-25T09:20:00Z"}
    assert failure_code(payload) == FailureCode.INVALID_TIMESTAMP


# V11, first half
def test_zone_intrusion_needs_zone_id() -> None:
    assert failure_code(detection("zone_intrusion")) == FailureCode.INVALID_ZONE


def test_zone_intrusion_with_zone_id_is_accepted() -> None:
    assert parse(detection("zone_intrusion", zone_id="zone-1")).zone_id == "zone-1"


# V12
def test_track_ids_are_unique_within_an_event() -> None:
    payload = srs_sample()
    payload["participants"][1]["track_id"] = 17
    assert failure_code(payload) == FailureCode.INVALID_PARTICIPANTS


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(detection("fall", participants=[]), id="fall"),
        pytest.param(srs_sample() | {"participants": []}, id="fight confirmation"),
    ],
)
def test_incident_creating_event_needs_a_participant(payload: dict[str, Any]) -> None:
    assert failure_code(payload) == FailureCode.INVALID_PARTICIPANTS


def test_fire_may_have_no_participants() -> None:
    assert parse(detection("fire", participants=[])).participants == []


def test_suspicion_may_have_no_participants() -> None:
    payload = srs_sample() | {"event_type": "suspicion", "participants": []}
    assert parse(payload).participants == []


# V14
STAGES = [("suspicion", "fight"), ("confirmation", "fight"), ("detection", "fall")]


@pytest.mark.parametrize(("event_type", "incident_type"), STAGES)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("embedding", [0.1] * 512, id="embedding"),
        pytest.param("embedding_model_version", "insightface-1.0", id="model version"),
        pytest.param("snapshot_ref", "snapshots/17.jpg", id="snapshot"),
        # Malformed face data is reported as face data too.
        pytest.param("embedding", "abc", id="embedding as string"),
        pytest.param("embedding", [], id="empty embedding"),
        pytest.param("embedding", [0.1, None, math.nan], id="odd embedding"),
        pytest.param("embedding", "a\x00", id="embedding with U+0000"),
        pytest.param("snapshot_ref", "", id="empty snapshot"),
        pytest.param("snapshot_ref", False, id="snapshot as bool"),
    ],
)
def test_face_data_is_rejected_in_any_event(
    event_type: str, incident_type: str, field: str, value: object
) -> None:
    payload = srs_sample() | {"event_type": event_type, "incident_type": incident_type}
    payload["participants"][1][field] = value
    assert failure_code(payload) == FailureCode.BIOMETRIC_NOT_ALLOWED


def test_null_face_fields_are_accepted() -> None:
    payload = srs_sample()
    payload["participants"][0] |= {
        "embedding": None,
        "embedding_model_version": None,
        "snapshot_ref": None,
    }
    assert parse(payload)
