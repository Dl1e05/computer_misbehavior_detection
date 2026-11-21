import json

import pytest

from app.services.ingest import DEAD_LETTER_PAYLOAD_MAX_BYTES, storable_payload


def test_plain_object_is_kept() -> None:
    assert storable_payload(b'{"camera_id": "cam-01", "n": 1}') == {
        "camera_id": "cam-01",
        "n": 1,
    }


def test_embeddings_are_removed_at_any_depth() -> None:
    raw = json.dumps(
        {
            "embedding": [0.1],
            "participants": [
                {"track_id": "1", "embedding": [0.2], "embedding_model_version": "v1"}
            ],
            "extra": {"deeper": [{"embedding": [0.3]}]},
        }
    ).encode()

    assert storable_payload(raw) == {
        "participants": [{"track_id": "1", "embedding_model_version": "v1"}],
        "extra": {"deeper": [{}]},
    }


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"not json", id="not json"),
        pytest.param(b"\xff\xfe{}", id="not utf-8"),
        pytest.param(b"[1, 2]", id="array"),
        pytest.param(b'"text"', id="string"),
        pytest.param(b'{"confidence": NaN}', id="NaN"),
        pytest.param(b'{"confidence": -Infinity}', id="Infinity"),
        pytest.param(b'{"name": "a\\u0000b"}', id="U+0000 in a value"),
        pytest.param(b'{"a\\u0000": 1}', id="U+0000 in a key"),
        pytest.param(b'{"name": "a\\ud800b"}', id="lone surrogate"),
        pytest.param(b'{"a": ' * 100_000 + b"1" + b"}" * 100_000, id="deep nesting"),
    ],
)
def test_payload_postgres_cannot_store_becomes_none(raw: bytes) -> None:
    assert storable_payload(raw) is None


def test_too_large_payload_becomes_none() -> None:
    raw = json.dumps({"padding": "x" * DEAD_LETTER_PAYLOAD_MAX_BYTES}).encode()
    assert storable_payload(raw) is None


def test_size_is_measured_without_embeddings() -> None:
    # A large embedding alone must not cost the rest of the payload.
    raw = json.dumps(
        {"camera_id": "cam-01", "embedding": [0.123456789] * 30_000}
    ).encode()
    assert len(raw) > DEAD_LETTER_PAYLOAD_MAX_BYTES
    assert storable_payload(raw) == {"camera_id": "cam-01"}
