"""Артефакт ТЗ 5.3 «Примеры событий»: JSON Schema и примеры валидных/невалидных сообщений для BE.

    python scripts/export_contract.py      # -> examples/events/{schema,valid,invalid}/

tests/test_contract.py проверяет: каждый valid/* проходит схему, каждый invalid/* — нет.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.events.schema import (CameraStatus, Event, EventParticipant, Heartbeat, Participant,  # noqa: E402
                                  json_schemas)

OUT = ROOT / "examples" / "events"
T = datetime(2026, 11, 21, 5, 15, 32, 480000, tzinfo=timezone.utc)
RUN = "ml-20261121-0501"
EV_FIGHT = uuid.UUID("0b7c2f0e-5a1d-4c4e-9a43-3f1b6f9d2a10")
EV_ZONE = uuid.UUID("5f0c3b52-8d0e-4f0b-9f5e-2b6c1a7d9e01")
MV = "ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/yolo26n-pose-1.0.0/fight-heur-0.1.0"
P17 = EventParticipant(track_id=17, bbox=(0.322, 0.167, 0.414, 0.639), confidence=0.88)
P21 = EventParticipant(track_id=21, bbox=(0.391, 0.181, 0.477, 0.653), confidence=0.83)


def unit_vector(seed: int) -> list[float]:
    v = np.random.default_rng(seed).normal(size=512)
    return [round(float(x), 6) for x in v / np.linalg.norm(v)]


def valid() -> dict[str, bytes]:
    fight = dict(event_type="suspicion", incident_type="fight", camera_id="cam-01", occurred_at=T,
                 confidence=0.56, participants=[P17, P21], model_version=MV, run_id=RUN,
                 episode_id="cam-01:20261121T051532:1")
    return {
        "event_zone_intrusion": Event(event_id=EV_ZONE, event_type="detection", incident_type="zone_intrusion",
                                      camera_id="cam-01", occurred_at=T, confidence=0.86, participants=[P17],
                                      model_version="ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/zone-0.1.0",
                                      run_id=RUN, zone_id="zone-3", zone_version=2),
        "event_fall": Event(event_id=uuid.UUID("9a1d7c55-0c41-4f3e-8e21-6d2b5a9f1c33"), event_type="detection",
                            incident_type="fall", camera_id="cam-02", occurred_at=T, confidence=0.9,
                            participants=[P21], run_id=RUN,
                            model_version="ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/yolo26n-pose-1.0.0/fall-rule-0.1.0"),
        "event_fight_suspicion": Event(event_id=uuid.UUID("7e3f9a10-2b6c-4d8e-a1f0-3c5b7d9e1f22"), **fight),
        "event_fight_confirmation": Event(event_id=EV_FIGHT, **{**fight, "event_type": "confirmation",
                                                               "confidence": 0.74,
                                                               "occurred_at": T.replace(second=34)}),
        "event_fire": Event(event_id=uuid.UUID("c2d4e6f8-1a3b-4c5d-8e7f-9a0b1c2d3e4f"), event_type="detection",
                            incident_type="fire", camera_id="cam-03", occurred_at=T, confidence=0.71,
                            participants=[], bbox=(0.41, 0.38, 0.52, 0.55), run_id=RUN,
                            model_version="ml-0.1.0/fire-yolo26n-1.0.0/fire-mofn-0.1.0"),
        "participant_face": Participant(event_id=EV_FIGHT, camera_id="cam-01", track_id=17,
                                        snapshot_ref=f"faces/cam-01/20261121/{EV_FIGHT}_17.jpg",
                                        snapshot_quality=0.78, embedding=unit_vector(17),
                                        embedding_model_version="insightface-buffalo_l-1.0"),
        "participant_no_usable_face": Participant(event_id=EV_FIGHT, camera_id="cam-01", track_id=21,
                                                  snapshot_quality="no_usable_face",
                                                  embedding_model_version="insightface-buffalo_l-1.0"),
        "heartbeat_online": Heartbeat(node_id="ml-node-1", run_id=RUN, status="online", sent_at=T,
                                      model_version="ml-0.1.0/yolo26n-det-1.0.0/yolo26n-pose-1.0.0/bytetrack-1.0",
                                      cameras=[CameraStatus(camera_id="cam-01", status="online",
                                                            rules={"zone_intrusion": "on", "fight": "on"}),
                                               CameraStatus(camera_id="cam-03", status="offline",
                                                            rules={"fire": "off: нет весов (обучение — Фаза 3.1)"})]),
        "heartbeat_last_will": Heartbeat(node_id="ml-node-1", run_id=RUN, status="offline"),
    }


def invalid(v: dict[str, dict]) -> dict[str, tuple[dict, str]]:
    """Невалидные: (json, что не так). Строятся из валидных с одной поломкой."""
    zone, fall, fight, fire = v["event_zone_intrusion"], v["event_fall"], v["event_fight_suspicion"], v["event_fire"]
    face, noface, hb = v["participant_face"], v["participant_no_usable_face"], v["heartbeat_online"]
    emb = face["embedding"]
    return {
        "event_fall_suspicion": ({**fall, "event_type": "suspicion"}, "suspicion/confirmation — только у fight"),
        "event_fight_detection": ({**fight, "event_type": "detection"}, "у fight только suspicion/confirmation"),
        "event_zone_without_zone_id": ({k: x for k, x in zone.items() if k != "zone_id"}, "zone_intrusion без zone_id"),
        "event_fight_without_episode_id": ({k: x for k, x in fight.items() if k != "episode_id"},
                                           "fight без episode_id"),
        "event_fire_without_bbox": ({k: x for k, x in fire.items() if k != "bbox"}, "fire без bbox очага"),
        "event_fire_with_participants": ({**fire, "participants": zone["participants"]}, "fire: participants пустой"),
        "event_bbox_on_fall": ({**fall, "bbox": [0.1, 0.1, 0.2, 0.2]}, "bbox события — только у fire"),
        "event_embedding_in_participant": ({**zone, "participants": [{**zone["participants"][0], "embedding": emb}]},
                                           "в Event нет данных лица"),
        "event_snapshot_in_participant": ({**zone, "participants": [{**zone["participants"][0],
                                                                      "snapshot_ref": "faces/x.jpg"}]},
                                          "в Event нет данных лица"),
        "event_bbox_out_of_range": ({**zone, "participants": [{**zone["participants"][0],
                                                                "bbox": [0.3, 0.1, 1.4, 0.6]}]}, "bbox — доли 0–1"),
        "event_model_version_too_long": ({**zone, "model_version": "x" * 129}, "model_version > 128"),
        "event_without_run_id": ({k: x for k, x in zone.items() if k != "run_id"}, "run_id обязателен"),
        "event_time_without_timezone": ({**zone, "occurred_at": "2026-11-21T05:15:32.480"}, "время без зоны"),
        "event_bad_event_id": ({**zone, "event_id": "123"}, "event_id — uuid"),
        "event_unknown_incident": ({**zone, "incident_type": "running"}, "incident_type не согласован"),
        "event_unknown_field": ({**zone, "severity": "high"}, "лишнее поле (Важность ставит BE)"),
        "participant_no_face_with_embedding": ({**noface, "embedding": emb}, "no_usable_face — без вектора"),
        "participant_face_without_ref": ({k: x for k, x in face.items() if k != "snapshot_ref"},
                                         "есть лицо — нужен snapshot_ref"),
        "participant_embedding_wrong_size": ({**face, "embedding": emb[:128]}, "вектор — 512 чисел"),
        "participant_embedding_not_normalized": ({**face, "embedding": [x * 2 for x in emb]}, "вектор длины 1"),
        "heartbeat_online_without_time": ({k: x for k, x in hb.items() if k != "sent_at"}, "online без sent_at"),
    }


def main() -> None:
    for sub in ("schema", "valid", "invalid"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for f in (OUT / sub).glob("*.json"):
            f.unlink()
    for kind, schema in json_schemas().items():
        (OUT / "schema" / f"{kind}.schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
    good = {name: json.loads(m.to_bytes()) for name, m in valid().items()}
    for name, d in good.items():
        (OUT / "valid" / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = invalid(good)
    for name, (d, why) in bad.items():
        (OUT / "invalid" / f"{name}.json").write_text(
            json.dumps({"_why_invalid": why, **d}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"схемы: {len(json_schemas())}, валидных: {len(good)}, невалидных: {len(bad)} -> {OUT}")


if __name__ == "__main__":
    main()
