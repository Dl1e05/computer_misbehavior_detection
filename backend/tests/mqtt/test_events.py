import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from app.domain.event import EventType, FailureCode
from app.domain.incident import IncidentType
from app.mqtt.events import EventLimits, MlEvent

DIM = 4  # a short embedding keeps the tests readable
LIMITS = EventLimits(
    embedding_dim=DIM,
    received_at=datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC),
    max_future=timedelta(seconds=60),
)


def srs_sample() -> dict[str, Any]:
    """The example event from SRS §7, as ML would publish it."""
    return {
        "schema_version": "1.0",
        "event_id": "5f0c2a7e-3b9d-4c1a-9e8f-2d7b6a1c0e42",
        "event_type": "confirmation",
        "camera_id": "cam-02",
        "occurred_at": "2026-10-25T09:14:03.217Z",
        "incident_type": "fight",
        "model_version": "fight-temporal-1.3.0/yolo26-det-2.0.1/bytetrack-1.0",
        "participants": [
            {
                "track_id": "17",
                "confidence": 0.91,
                "bbox": [412.0, 188.5, 498.0, 402.0],
                "snapshot_ref": "snapshots/2026/10/25/5f0c2a7e-17.jpg",
            },
            {
                "track_id": "23",
                "confidence": 0.88,
                "bbox": [520.0, 175.0, 604.0, 395.0],
                "snapshot_ref": "snapshots/2026/10/25/5f0c2a7e-23.jpg",
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


def with_embedding(payload: dict[str, Any], **changes: Any) -> dict[str, Any]:
    payload["participants"][0] |= {
        "embedding": [0.1] * DIM,
        "embedding_model_version": "insightface-buffalo_l-1.0",
        **changes,
    }
    return payload


def test_srs_sample_parses() -> None:
    event = parse(srs_sample())

    assert event.event_type is EventType.CONFIRMATION
    assert event.incident_type is IncidentType.FIGHT
    assert event.occurred_at.tzinfo is not None
    assert [p.track_id for p in event.participants] == ["17", "23"]
    assert event.participants[0].bbox == [412.0, 188.5, 498.0, 402.0]
    assert event.zone_id is None
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
    payload = srs_sample() | {"run_id": "run-7"}
    payload["participants"][0]["new_field"] = 1

    event = parse(payload)
    assert not hasattr(event, "run_id")


def test_episode_id_is_kept() -> None:
    event = parse(srs_sample() | {"episode_id": "ep-42"})
    assert event.episode_id == "ep-42"


def test_whole_numbers_are_accepted_in_bbox() -> None:
    payload = srs_sample()
    payload["participants"][0]["bbox"] = [412, 188, 498, 402]
    assert parse(payload).participants[0].bbox == [412.0, 188.0, 498.0, 402.0]


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
    ],
)
def test_invalid_event_field_is_rejected(field: str, value: object) -> None:
    payload = srs_sample() | {field: value}
    with pytest.raises(ValidationError) as error:
        parse(payload)
    assert error.value.errors()[0]["loc"][0] == field


@pytest.mark.parametrize(
    "required", ["schema_version", "event_id", "event_type", "camera_id"]
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
        pytest.param("bbox", [1.0, 2.0, 3.0], id="bbox with 3 numbers"),
        pytest.param("bbox", [1.0, 2.0, 3.0, 4.0, 5.0], id="bbox with 5 numbers"),
        pytest.param("track_id", "", id="empty track_id"),
        pytest.param("track_id", 17, id="track_id as number"),
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
    # Without them V9 and V15 would be skipped silently, so it is an error.
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
    payload["participants"][1]["track_id"] = "17"
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


# V14 and V15
def test_embedding_on_confirmation_is_accepted() -> None:
    event = parse(with_embedding(srs_sample()))
    assert event.participants[0].embedding == [0.1] * DIM


def test_embedding_on_suspicion_is_rejected() -> None:
    payload = with_embedding(srs_sample() | {"event_type": "suspicion"})
    assert failure_code(payload) == FailureCode.BIOMETRIC_NOT_ALLOWED


def test_embedding_needs_its_model_version() -> None:
    payload = with_embedding(srs_sample(), embedding_model_version=None)
    assert failure_code(payload) == FailureCode.MISSING_FIELD


@pytest.mark.parametrize("length", [DIM - 1, DIM + 1, 0])
def test_embedding_must_have_dim_numbers(length: int) -> None:
    payload = with_embedding(srs_sample(), embedding=[0.1] * length)
    assert failure_code(payload) == FailureCode.INVALID_EMBEDDING


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    # json.dumps writes NaN / Infinity, which Python's json would read back.
    payload = with_embedding(srs_sample(), embedding=[0.1, 0.2, 0.3, value])
    with pytest.raises(ValidationError, match="finite"):
        parse(payload)
