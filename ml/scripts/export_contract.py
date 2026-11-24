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
RUN = "ml-node-1-20261121T050112Z-7f3a"       # <node_id>-<UTC>-<4 hex>
ZONE_3 = uuid.UUID("2f6e1c9a-4b7d-4a3e-8c12-9d0e5f7a1b34")   # UUID зон/линий — как из cameras-zones
ZONE_HALL = uuid.UUID("8a4d2b6e-1f3c-4e9a-b7d5-0c2e4f6a8b91")
LINE_DOOR = uuid.UUID("c3e5a7b9-2d4f-4061-8a3c-5e7f9b1d3a52")
EV_FIGHT = uuid.UUID("0b7c2f0e-5a1d-4c4e-9a43-3f1b6f9d2a10")
EV_ZONE = uuid.UUID("5f0c3b52-8d0e-4f0b-9f5e-2b6c1a7d9e01")
MV = "ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/yolo26n-pose-1.0.0/fight-heur-0.1.0"
P17 = EventParticipant(track_id=17, bbox=(0.322, 0.167, 0.414, 0.639), confidence=0.88)
P21 = EventParticipant(track_id=21, bbox=(0.391, 0.181, 0.477, 0.653), confidence=0.83)


def unit_vector(seed: int) -> list[float]:
    """СИНТЕТИКА: случайный единичный вектор с фиксированным seed — не из лица (можно в тесты BE)."""
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
                                      run_id=RUN, zone_id=ZONE_3, zone_version=2),
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
                                        snapshot_quality="ok", quality_score=0.78, embedding=unit_vector(17),
                                        embedding_model_version="insightface-buffalo_l-1.0"),
        "participant_no_usable_face": Participant(event_id=EV_FIGHT, camera_id="cam-01", track_id=21,
                                                  snapshot_quality="no_usable_face",
                                                  embedding_model_version="insightface-buffalo_l-1.0"),
        "heartbeat_online": Heartbeat(node_id="ml-node-1", run_id=RUN, status="online", sent_at=T,
                                      model_version="ml-0.1.0/yolo26n-det-1.0.0/yolo26n-pose-1.0.0/bytetrack-1.0",
                                      cameras=[CameraStatus(camera_id="cam-01", status="online",
                                                            rules={"zone_intrusion": "on", "fight": "on"}),
                                               CameraStatus(camera_id="cam-03", status="offline",
                                                            rules={"fire": "off: нет весов (обучение — Фаза 3.1)"})],
                                      dropped_stale=0),
        "heartbeat_last_will": Heartbeat(node_id="ml-node-1", run_id=RUN, status="offline"),
        # --- schema 1.1: предложено BE, ждёт одобрения (ML не отправляет их в MQTT); details — своя схема на тип ---
        "event_proposed_line_crossing": Event(
            schema_version="1.1", event_id=uuid.UUID("3a1f0c2e-6b7d-4e8f-9a0b-1c2d3e4f5a61"), event_type="detection",
            incident_type="line_crossing", camera_id="cam-01", occurred_at=T, confidence=0.88, participants=[P17],
            model_version="ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/line-0.1.0", run_id=RUN,
            line_id=LINE_DOOR, details={"direction": "in"}),
        "event_proposed_loitering": Event(
            schema_version="1.1", event_id=uuid.UUID("4b2a1d3f-7c8e-4f90-8b1c-2d3e4f5a6b72"), event_type="detection",
            incident_type="loitering", camera_id="cam-01", occurred_at=T, confidence=0.83, participants=[P21],
            model_version="ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/loiter-0.1.0", run_id=RUN,
            zone_id=ZONE_3, zone_version=2, details={"dwell_s": 64.5}),
        "event_proposed_crowding": Event(
            schema_version="1.1", event_id=uuid.UUID("5c3b2e4a-8d9f-4a01-9c2d-3e4f5a6b7c83"), event_type="detection",
            incident_type="crowding", camera_id="cam-01", occurred_at=T, confidence=0.75, participants=[P17, P21],
            model_version="ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/crowd-0.1.0", run_id=RUN,
            zone_id=ZONE_HALL, zone_version=1, details={"count": 9}),
        "event_proposed_camera_tampering": Event(
            schema_version="1.1", event_id=uuid.UUID("6d4c3f5b-9e0a-4b12-8d3e-4f5a6b7c8d94"), event_type="detection",
            incident_type="camera_tampering", camera_id="cam-02", occurred_at=T, confidence=0.93, participants=[],
            model_version="ml-0.1.0/tamper-0.1.0", run_id=RUN, details={"tamper_type": "covered"}),
        "event_proposed_abandoned_object": Event(
            schema_version="1.1", event_id=uuid.UUID("8f6e5b7d-1a2c-4d34-8f5a-6b7c8d9eafb6"), event_type="detection",
            incident_type="abandoned_object", camera_id="cam-01", occurred_at=T, confidence=0.62, participants=[],
            bbox=(0.47, 0.64, 0.53, 0.70), model_version="ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/objects-0.1.0",
            run_id=RUN, details={"object_class": "backpack", "unattended_s": 31.5}),
        "event_proposed_object_taken": Event(
            schema_version="1.1", event_id=uuid.UUID("9a7f6c8e-2b3d-4e45-9a6b-7c8d9eafb0c7"), event_type="detection",
            incident_type="object_taken", camera_id="cam-01", occurred_at=T, confidence=0.7, participants=[P21],
            bbox=(0.47, 0.64, 0.53, 0.70), model_version="ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/objects-0.1.0",
            run_id=RUN, details={"object_class": "backpack"}),
        "event_proposed_scene_change": Event(
            schema_version="1.1", event_id=uuid.UUID("ab8f7d9e-3c4e-4f56-8b7c-8d9eafb0c1d8"), event_type="detection",
            incident_type="scene_change", camera_id="cam-04", occurred_at=T, confidence=0.9, participants=[P17],
            model_version="ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/scene-0.1.0", run_id=RUN,
            zone_id=ZONE_3, zone_version=1, details={"change_ratio": 0.52, "seconds_since_person": 4.0}),
        "event_proposed_smoke": Event(
            schema_version="1.1", event_id=uuid.UUID("7e5d4a6c-0f1b-4c23-9e4f-5a6b7c8d9ea5"), event_type="detection",
            incident_type="smoke", camera_id="cam-03", occurred_at=T, confidence=0.66, participants=[],
            bbox=(0.30, 0.10, 0.62, 0.42), model_version="ml-0.1.0/fire-yolo26n-1.0.0/smoke-mofn-0.1.0", run_id=RUN),
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
        "event_unknown_incident": ({**zone, "incident_type": "dancing"}, "incident_type не согласован и не предложен"),
        "event_proposed_with_schema_1_0": ({**v["event_proposed_line_crossing"], "schema_version": "1.0"},
                                           "новый тип — только schema 1.1"),
        "event_line_without_direction": ({**v["event_proposed_line_crossing"], "details": {}},
                                         "line_crossing: details.direction обязателен"),
        "event_line_without_line_id": ({k: x for k, x in v["event_proposed_line_crossing"].items()
                                        if k != "line_id"}, "line_crossing: line_id обязателен"),
        "event_details_of_wrong_type": ({**v["event_proposed_loitering"], "details": {"count": 3}},
                                        "у loitering details — {dwell_s}, а не {count}"),
        "event_details_on_old_type": ({**zone, "details": {"dwell_s": 3}}, "details — только у новых типов (1.1)"),
        "event_flat_field_instead_of_details": ({**v["event_proposed_crowding"], "count": 9},
                                                "поля типа — только внутри details"),
        "event_zone_id_not_uuid": ({**zone, "zone_id": "zone-3"}, "zone_id — UUID из cameras-zones"),
        "event_zone_without_zone_version": ({k: x for k, x in zone.items() if k != "zone_version"},
                                            "zone_intrusion без zone_version"),
        "event_smoke_without_bbox": ({k: x for k, x in v["event_proposed_smoke"].items() if k != "bbox"},
                                     "smoke без bbox очага"),
        "event_scene_change_without_zone": ({k: x for k, x in v["event_proposed_scene_change"].items()
                                              if k not in ("zone_id", "zone_version")},
                                             "scene_change: нужна охраняемая зона"),
        "event_scene_change_ratio_over_1": ({**v["event_proposed_scene_change"],
                                             "details": {"change_ratio": 1.5, "seconds_since_person": 4.0}},
                                            "change_ratio — доля 0–1"),
        "event_object_taken_without_taker": ({**v["event_proposed_object_taken"], "participants": []},
                                             "object_taken: нужен тот, кто унёс"),
        "event_abandoned_without_object_bbox": ({k: x for k, x in v["event_proposed_abandoned_object"].items()
                                                 if k != "bbox"}, "abandoned_object: нужен bbox предмета"),
        "event_abandoned_unknown_object_class": ({**v["event_proposed_abandoned_object"],
                                                  "details": {"object_class": "phone", "unattended_s": 30}},
                                                 "object_class — из списка классов"),
        "event_tampering_with_participants": ({**v["event_proposed_camera_tampering"],
                                               "participants": zone["participants"]},
                                              "camera_tampering: participants пустой"),
        "event_unknown_field": ({**zone, "severity": "high"}, "лишнее поле (Важность ставит BE)"),
        "participant_no_face_with_embedding": ({**noface, "embedding": emb}, "no_usable_face — без вектора"),
        "participant_face_without_ref": ({k: x for k, x in face.items() if k != "snapshot_ref"},
                                         "ok — нужен snapshot_ref"),
        "participant_ok_without_quality_score": ({k: x for k, x in face.items() if k != "quality_score"},
                                                 "ok — нужен quality_score"),
        "participant_quality_as_number": ({**face, "snapshot_quality": 0.78}, "snapshot_quality — ok | no_usable_face"),
        "participant_ref_of_other_track": ({**face, "snapshot_ref": face["snapshot_ref"].replace("_17.jpg", "_21.jpg")},
                                           "snapshot_ref не от этого event_id/track_id"),
        "participant_without_schema_version": ({k: x for k, x in face.items() if k != "schema_version"},
                                               "schema_version обязателен"),
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
