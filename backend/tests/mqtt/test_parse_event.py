import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.domain.event import FailureCode
from app.mqtt.events import EventLimits, MlEvent, Rejection, parse_event

DIM = 4
LIMITS = EventLimits(
    embedding_dim=DIM,
    received_at=datetime(2026, 10, 25, 9, 14, 4, tzinfo=UTC),
    max_future=timedelta(seconds=60),
)


def valid_event() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "event_id": "5f0c2a7e-3b9d-4c1a-9e8f-2d7b6a1c0e42",
        "event_type": "detection",
        "camera_id": "cam-02",
        "occurred_at": "2026-10-25T09:14:03.217Z",
        "incident_type": "fall",
        "model_version": "fall-1.0",
        "participants": [{"track_id": "17", "confidence": 0.91}],
    }


def rejected(raw: bytes) -> Rejection:
    result = parse_event(raw, LIMITS)
    assert isinstance(result, Rejection), result
    return result


def code_for(payload: dict[str, Any]) -> FailureCode:
    return rejected(json.dumps(payload).encode()).code


def test_valid_event_is_returned() -> None:
    result = parse_event(json.dumps(valid_event()).encode(), LIMITS)
    assert isinstance(result, MlEvent)
    assert result.camera_id == "cam-02"


# V1
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"not json", id="not json"),
        pytest.param(b"", id="empty"),
        pytest.param(b'{"schema_version": "1.0"', id="cut off"),
        pytest.param(b"\xff\xfe{}", id="not utf-8"),
        pytest.param(b"[1, 2]", id="array"),
        pytest.param(b'"text"', id="string"),
    ],
)
def test_malformed_message(raw: bytes) -> None:
    assert rejected(raw).code is FailureCode.MALFORMED_JSON


# V2
def test_missing_field() -> None:
    payload = valid_event()
    del payload["camera_id"]
    assert code_for(payload) is FailureCode.MISSING_FIELD


@pytest.mark.parametrize("field", ["event_id", "event_type", "occurred_at"])
def test_null_counts_as_missing(field: str) -> None:
    # "present and non-null": a null would otherwise be reported as a bad type.
    assert code_for(valid_event() | {field: None}) is FailureCode.MISSING_FIELD


def test_null_participant_confidence_counts_as_missing() -> None:
    payload = valid_event()
    payload["participants"][0]["confidence"] = None
    assert code_for(payload) is FailureCode.MISSING_FIELD


# One code per field rule, from field errors (not our custom rule errors).
@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"schema_version": 1}, FailureCode.UNSUPPORTED_SCHEMA),
        ({"event_id": "not-a-uuid"}, FailureCode.INVALID_EVENT_ID),
        ({"event_type": "telepathy"}, FailureCode.INVALID_EVENT_TYPE),
        ({"incident_type": "theft"}, FailureCode.INVALID_INCIDENT_TYPE),
        ({"camera_id": ""}, FailureCode.UNKNOWN_CAMERA),
        ({"occurred_at": "yesterday"}, FailureCode.INVALID_TIMESTAMP),
        ({"sent_at": "yesterday"}, FailureCode.INVALID_TIMESTAMP),
        ({"zone_id": 7}, FailureCode.INVALID_ZONE),
        ({"participants": "17"}, FailureCode.INVALID_PARTICIPANTS),
        ({"model_version": "x" * 129}, FailureCode.INVALID_FIELD),
        ({"episode_id": 5}, FailureCode.INVALID_FIELD),
        # null in an optional field is that field's error, not "missing".
        ({"participants": None}, FailureCode.INVALID_PARTICIPANTS),
        ({"zone_id": None, "sent_at": None}, None),
    ],
)
def test_field_error_codes(
    changes: dict[str, Any], expected: FailureCode | None
) -> None:
    payload = valid_event() | changes
    if expected is None:
        assert isinstance(parse_event(json.dumps(payload).encode(), LIMITS), MlEvent)
    else:
        assert code_for(payload) is expected


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"confidence": 1.5}, FailureCode.INVALID_CONFIDENCE),
        ({"confidence": "0.9"}, FailureCode.INVALID_CONFIDENCE),
        ({"track_id": ""}, FailureCode.INVALID_PARTICIPANTS),
        ({"bbox": [1.0, 2.0]}, FailureCode.INVALID_PARTICIPANTS),
        ({"embedding": "abc"}, FailureCode.INVALID_EMBEDDING),
        # A null inside a list is that list's error, not a missing field.
        (
            {"embedding": [1.0, 2.0, None, 4.0], "embedding_model_version": "v1"},
            FailureCode.INVALID_EMBEDDING,
        ),
        ({"bbox": [None, 1.0, 2.0, 3.0]}, FailureCode.INVALID_PARTICIPANTS),
    ],
)
def test_participant_error_codes(
    changes: dict[str, Any], expected: FailureCode
) -> None:
    payload = valid_event()
    payload["participants"][0] |= changes
    assert code_for(payload) is expected


def test_rule_error_codes_pass_through() -> None:
    # A custom rule error (V7) arrives with its code as the error type.
    payload = valid_event() | {"incident_type": "fight"}
    assert code_for(payload) is FailureCode.INVALID_STAGE_TYPE_PAIR


# First failure by V order
@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        pytest.param(
            {"camera_id": None, "event_id": "bad"},
            FailureCode.MISSING_FIELD,
            id="V2 before V4",
        ),
        pytest.param(
            {"schema_version": "9.0", "event_id": "bad", "incident_type": "theft"},
            FailureCode.UNSUPPORTED_SCHEMA,
            id="V3 before V4 and V6",
        ),
        pytest.param(
            {"event_type": "telepathy", "occurred_at": "yesterday"},
            FailureCode.INVALID_EVENT_TYPE,
            id="V5 before V9",
        ),
        pytest.param(
            {"incident_type": "theft", "participants": [{"confidence": 2}]},
            FailureCode.MISSING_FIELD,
            id="V2 in a participant before V6",
        ),
    ],
)
def test_first_failure_in_v_order_wins(
    changes: dict[str, Any], expected: FailureCode
) -> None:
    assert code_for(valid_event() | changes) is expected


def test_detail_names_the_field() -> None:
    payload = valid_event()
    payload["participants"][0]["confidence"] = 1.5
    detail = rejected(json.dumps(payload).encode()).detail
    assert detail.startswith("participants.0.confidence:")


def test_detail_never_contains_input_values() -> None:
    # An embedding must never reach logs or dead letters (SRS §14).
    secret = 0.123456789
    payload = valid_event() | {"event_type": "suspicion", "incident_type": "fight"}
    payload["participants"][0] |= {
        "embedding": [secret] * DIM,
        "embedding_model_version": "insightface-1.0",
    }
    rejection = rejected(json.dumps(payload).encode())

    assert rejection.code is FailureCode.BIOMETRIC_NOT_ALLOWED
    assert str(secret) not in rejection.detail
    assert str(secret) not in repr(rejection)


def test_detail_of_malformed_message_has_no_payload() -> None:
    raw = b'{"embedding": [0.123456789, 0.987654321'
    rejection = rejected(raw)
    assert "0.123456789" not in rejection.detail


def test_invalid_field_never_hides_a_rule_failure() -> None:
    # INVALID_FIELD ranks after every V code.
    payload = valid_event() | {"episode_id": 5, "event_type": "telepathy"}
    assert code_for(payload) is FailureCode.INVALID_EVENT_TYPE


def test_known_deviation_cross_field_rule_next_to_field_error() -> None:
    # Pydantic runs cross-field rules only when every field is valid, so a
    # V7 fault next to a V13 field error is reported as V13 (see
    # MlEvent._cross_field_rules). Pinned here so a change is noticed.
    payload = valid_event() | {"incident_type": "fight"}
    payload["participants"][0]["confidence"] = 1.5
    assert code_for(payload) is FailureCode.INVALID_CONFIDENCE


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("camera_id", FailureCode.UNKNOWN_CAMERA),
        ("model_version", FailureCode.INVALID_FIELD),
        ("zone_id", FailureCode.INVALID_ZONE),
    ],
)
def test_nul_in_text_is_rejected(field: str, expected: FailureCode) -> None:
    # Postgres cannot store U+0000; saving it would fail on every redelivery.
    assert code_for(valid_event() | {field: "a\x00b"}) is expected


def test_nul_in_participant_text_is_rejected() -> None:
    payload = valid_event()
    payload["participants"][0]["track_id"] = "17\x00"
    assert code_for(payload) is FailureCode.INVALID_PARTICIPANTS


@pytest.mark.parametrize("version", ["1.0\n", "\u0661.0", " 1.0"])
def test_schema_version_must_be_plain_ascii(version: str) -> None:
    payload = valid_event() | {"schema_version": version}
    assert code_for(payload) is FailureCode.UNSUPPORTED_SCHEMA


def test_limits_need_an_aware_received_at() -> None:
    # A naive time would crash later, and only on valid events.
    with pytest.raises(ValueError, match="timezone"):
        EventLimits(
            embedding_dim=DIM,
            received_at=datetime(2026, 10, 25, 9, 14, 4),
            max_future=timedelta(seconds=60),
        )


@pytest.mark.parametrize("raw", [b"not json", json.dumps(valid_event()).encode()])
def test_missing_limits_fail_for_every_message(raw: bytes) -> None:
    # Also for malformed messages, where the model's own check never runs.
    with pytest.raises(TypeError, match="EventLimits"):
        parse_event(raw, None)  # type: ignore[arg-type]
