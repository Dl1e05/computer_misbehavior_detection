"""Шаг 8: контракт ML -> BE, outbox и Publisher.

Без брокера: схема на примерах, сборка Event из Detection, outbox, Publisher на фальшивом клиенте.
С брокером (локальный Mosquitto на 1883, иначе skip): доставка тех же байтов, outbox при
недоступном брокере, Last Will при падении процесса.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from vision.events.builder import ModelVersions, event_from_detection, new_run_id, participant_from
from vision.events.outbox import Outbox
from vision.events.publisher import Publisher
from vision.events.schema import (MAX_BYTES, Event, Heartbeat, Participant, message_kind, parse, topic_event,
                                  topic_heartbeat, topic_participant)
from vision.face.participants import ParticipantMessage
from vision.rules.base import Detection, Participant as RuleParticipant

ML = Path(__file__).resolve().parents[1]
EXAMPLES = ML / "examples" / "events"
T = datetime(2026, 11, 24, 10, 0, 0, 123456, tzinfo=timezone.utc)


# ---------- схема на примерах (артефакт для BE) ----------

def _examples(sub):
    files = sorted((EXAMPLES / sub).glob("*.json"))
    assert files, f"нет примеров в {sub}: python scripts/export_contract.py"
    return files


def _kind(name: str) -> str:
    return name.split("_")[0]


@pytest.mark.parametrize("path", _examples("valid"), ids=lambda p: p.stem)
def test_valid_examples_pass(path):
    parse(_kind(path.stem), path.read_bytes())


@pytest.mark.parametrize("path", _examples("invalid"), ids=lambda p: p.stem)
def test_invalid_examples_fail(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    d.pop("_why_invalid")
    with pytest.raises(ValidationError):
        parse(_kind(path.stem), json.dumps(d))


def test_schema_files_exist_for_be():
    assert {p.name for p in (EXAMPLES / "schema").glob("*.json")} == {
        "event.schema.json", "participant.schema.json", "heartbeat.schema.json"}


def test_time_is_utc_with_ms_and_z():
    e = parse("event", (EXAMPLES / "valid" / "event_zone_intrusion.json").read_bytes())
    msk = datetime(2026, 11, 24, 13, 0, 0, 5000, tzinfo=timezone.utc).astimezone()   # любая зона -> UTC
    d = json.loads(e.model_copy(update={"occurred_at": msk}).to_bytes())
    assert d["occurred_at"] == "2026-11-24T13:00:00.005Z"


def test_bytes_are_stable_roundtrip():
    raw = (EXAMPLES / "valid" / "event_fight_confirmation.json").read_bytes()
    e = parse("event", raw)
    assert e.to_bytes() == parse("event", e.to_bytes()).to_bytes()


def test_size_limit():
    e = parse("event", (EXAMPLES / "valid" / "event_zone_intrusion.json").read_bytes())
    big = e.model_copy(update={"participants": e.participants * 6000})
    with pytest.raises(ValueError):
        big.to_bytes()
    assert len(e.to_bytes()) < MAX_BYTES


def test_topics():
    assert topic_event("cam-01") == "videoanalytics/v1/cameras/cam-01/events"
    assert topic_participant("cam-01") == "videoanalytics/v1/cameras/cam-01/participants"
    assert topic_heartbeat("ml-node-1") == "videoanalytics/v1/ml/ml-node-1/heartbeat"
    assert message_kind(topic_heartbeat("x")) == "heartbeat"


# ---------- Detection -> Event ----------

VERSIONS = ModelVersions({"ml": "0.1.0", "detector": "yolo26n-det-1.0.0", "pose": "yolo26n-pose-1.0.0",
                          "tracker": "bytetrack-1.0", "fire": "fire-yolo26n-0.0.0",
                          "rules": {"fight": "fight-heur-0.1.0", "fire": "fire-mofn-0.1.0"}})


def test_event_from_rule_detection():
    d = Detection("fight", "confirmation", "cam-01", T, 0.74,
                  [RuleParticipant(3, (0.1, 0.2, 0.3, 0.9), 0.91)], episode_id="cam-01:x:1")
    run = new_run_id(T)
    e = event_from_detection(d, run, VERSIONS.for_incident("fight", {"persons", "poses"}))
    assert run == "ml-20261124-100000"
    out = json.loads(e.to_bytes())
    assert out["model_version"] == "ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/yolo26n-pose-1.0.0/fight-heur-0.1.0"
    assert out["occurred_at"] == "2026-11-24T10:00:00.123Z"
    assert out["participants"] == [{"track_id": 3, "bbox": [0.1, 0.2, 0.3, 0.9], "confidence": 0.91}]
    assert "bbox" not in out and "zone_id" not in out          # None не отправляем
    uuid.UUID(out["event_id"])


def test_fire_event_and_version():
    d = Detection("fire", "detection", "cam-03", T, 0.7, [], bbox=(0.4, 0.4, 0.5, 0.52))
    e = event_from_detection(d, "ml-1", VERSIONS.for_incident("fire", {"fire"}))
    assert e.bbox == (0.4, 0.4, 0.5, 0.52) and e.participants == []
    assert e.model_version == "ml-0.1.0/fire-yolo26n-0.0.0/fire-mofn-0.1.0"


def test_participant_message_from_face_collector():
    eid = str(uuid.uuid4())
    m = ParticipantMessage(eid, "cam-01", 4, None, "no_usable_face", None, "insightface-buffalo_l-1.0", "no_face")
    p = participant_from(m)
    assert json.loads(p.to_bytes()) == {"schema_version": "1.0", "event_id": eid, "camera_id": "cam-01",
                                        "track_id": 4, "snapshot_quality": "no_usable_face",
                                        "embedding_model_version": "insightface-buffalo_l-1.0"}


# ---------- outbox ----------

def test_outbox_survives_restart_keeps_bytes_order_and_dedupes(tmp_path):
    path = tmp_path / "outbox.sqlite"
    ob = Outbox(path)
    assert ob.add("t/1", b"\x00first", "e1") and ob.add("t/2", b"second", "e2")
    assert not ob.add("t/1", b"other bytes", "e1")              # тот же ключ — дубль не встаёт
    ob.close()
    ob = Outbox(path)                                          # «перезапуск процесса»
    items = ob.pending()
    assert [(i.topic, i.payload, i.key) for i in items] == [("t/1", b"\x00first", "e1"), ("t/2", b"second", "e2")]
    ob.delete(items[0].id)
    assert [i.key for i in ob.pending()] == ["e2"] and len(ob) == 1


# ---------- Publisher на фальшивом клиенте ----------

class FakeInfo:
    def __init__(self, mid, rc=0):
        self.mid, self.rc = mid, rc

    def wait_for_publish(self, timeout=None):
        pass


class FakeClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.published, self.will, self.mid = [], None, 0
        self.up = False
        self.ack_before_return = False

    def username_pw_set(self, *a): pass
    def tls_set(self, *a): pass
    def reconnect_delay_set(self, **kw): pass
    def connect_async(self, *a, **kw): pass
    def loop_start(self): pass
    def loop_stop(self): pass
    def disconnect(self): pass

    def will_set(self, topic, payload, qos, retain):
        self.will = (topic, payload, qos, retain)

    def publish(self, topic, payload, qos, retain):
        assert qos == 1 and retain is False
        if not self.up:
            return FakeInfo(0, rc=4)
        self.mid += 1
        self.published.append((self.mid, topic, payload))
        if self.ack_before_return:                             # гонка: PUBACK раньше, чем publish вернулся
            self.on_publish(self, None, self.mid, 0, None)
        return FakeInfo(self.mid)

    # управление из теста
    def connect(self):
        self.up = True
        self.on_connect(self, None, {}, 0, None)

    def drop(self):
        self.up = False
        self.on_disconnect(self, None, {}, 7, None)

    def ack(self, mid):
        self.on_publish(self, None, mid, 0, None)


def make_pub(tmp_path, **kw):
    clients = []

    def factory(cid):
        clients.append(FakeClient(cid))
        return clients[-1]

    pub = Publisher("ml-node-1", "ml-run", Outbox(tmp_path / "ob.sqlite"), heartbeat_s=3600,
                    status_fn=lambda: {"cameras": [{"camera_id": "cam-01", "status": "online", "rules": {"fall": "on"}}]},
                    client_factory=factory, **kw)
    return pub, clients[0]


def wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def zone_event():
    return parse("event", (EXAMPLES / "valid" / "event_zone_intrusion.json").read_bytes()).model_copy(
        update={"event_id": uuid.uuid4()})


def test_last_will_is_offline_heartbeat_not_retained(tmp_path):
    pub, c = make_pub(tmp_path)
    topic, payload, qos, retain = c.will
    assert topic == "videoanalytics/v1/ml/ml-node-1/heartbeat" and qos == 1 and retain is False
    hb = parse("heartbeat", payload)
    assert hb.status == "offline" and hb.run_id == "ml-run"


def test_events_wait_offline_then_go_same_bytes_and_deleted_after_puback(tmp_path):
    pub, c = make_pub(tmp_path)
    pub.start()
    e = zone_event()
    pub.submit_event(e)
    time.sleep(0.3)
    assert c.published == [] and len(pub.outbox) == 1           # брокера нет — ждёт в outbox
    c.connect()
    assert wait(lambda: any(t.endswith("/events") for _, t, _ in c.published))
    mid, topic, payload = next(x for x in c.published if x[1].endswith("/events"))
    assert topic == "videoanalytics/v1/cameras/cam-01/events" and payload == e.to_bytes()
    assert len(pub.outbox) == 1                                 # без PUBACK не удаляем
    c.ack(mid)
    assert len(pub.outbox) == 0 and pub.sent == 1
    pub.stop(flush_timeout=0)


def test_reconnect_resends_unacked_with_same_bytes(tmp_path):
    pub, c = make_pub(tmp_path)
    pub.start()
    c.connect()
    e = zone_event()
    pub.submit_event(e)
    assert wait(lambda: any(t.endswith("/events") for _, t, _ in c.published))
    c.drop()                                                    # связь пропала до PUBACK
    c.published.clear()
    c.connect()
    assert wait(lambda: any(t.endswith("/events") for _, t, _ in c.published))
    mid, _, payload = next(x for x in c.published if x[1].endswith("/events"))
    assert payload == e.to_bytes()
    c.ack(mid)
    assert len(pub.outbox) == 0
    pub.stop(flush_timeout=0)


def test_puback_before_publish_returns_is_not_lost(tmp_path):
    pub, c = make_pub(tmp_path)
    c.ack_before_return = True
    pub.start()
    c.connect()
    pub.submit_event(zone_event())
    assert wait(lambda: len(pub.outbox) == 0)
    pub.stop(flush_timeout=0)


def test_heartbeat_has_camera_statuses(tmp_path):
    pub, c = make_pub(tmp_path)
    pub.start()
    c.connect()
    assert wait(lambda: any(t.endswith("/heartbeat") for _, t, _ in c.published))
    hb = parse("heartbeat", next(p for _, t, p in c.published if t.endswith("/heartbeat")))
    assert hb.status == "online" and hb.cameras[0].rules == {"fall": "on"}
    pub.stop(flush_timeout=0)
    assert parse("heartbeat", c.published[-1][2]).status == "offline"   # штатная остановка


def test_one_participant_message_per_event_and_track(tmp_path):
    pub, c = make_pub(tmp_path)
    eid = str(uuid.uuid4())
    m = participant_from(ParticipantMessage(eid, "cam-01", 4, None, "no_usable_face", None, "v1"))
    assert pub.submit_participant(m) and not pub.submit_participant(m)


# ---------- с настоящим Mosquitto ----------

def _broker_up(port=1883) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


needs_broker = pytest.mark.skipif(not _broker_up(), reason="нет Mosquitto на localhost:1883")


class Listener:
    def __init__(self, topic):
        import paho.mqtt.client as mqtt

        self.msgs, self.ready = [], threading.Event()
        self.c = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                             client_id=f"test-listen-{uuid.uuid4().hex[:8]}")
        self.c.on_connect = lambda c, u, f, rc, p=None: c.subscribe(topic, qos=1)
        self.c.on_subscribe = lambda *a: self.ready.set()
        self.c.on_message = lambda c, u, m: self.msgs.append((m.topic, m.payload))
        self.c.connect("localhost", 1883)
        self.c.loop_start()
        assert self.ready.wait(3)

    def close(self):
        self.c.disconnect()
        self.c.loop_stop()


@needs_broker
def test_real_broker_outbox_then_delivery_same_event_id(tmp_path):
    node = f"test-node-{uuid.uuid4().hex[:6]}"
    lst = Listener("videoanalytics/v1/cameras/cam-01/events")
    path = tmp_path / "ob.sqlite"
    e = zone_event()
    dead = Publisher(node, "run-1", Outbox(path), port=1, heartbeat_s=3600).start()   # брокер «лежит»
    dead.submit_event(e)
    time.sleep(1.0)
    assert len(dead.outbox) == 1 and not lst.msgs
    dead.stop(flush_timeout=0)
    dead.outbox.close()

    live = Publisher(node, "run-2", Outbox(path), heartbeat_s=3600).start()           # «перезапуск»
    assert wait(lambda: lst.msgs, timeout=5)
    assert wait(lambda: len(live.outbox) == 0, timeout=3)
    live.stop()
    lst.close()
    topic, payload = lst.msgs[0]
    assert payload == e.to_bytes()                              # те же байты, тот же event_id
    assert len(lst.msgs) == 1


@needs_broker
def test_real_broker_last_will_when_process_killed(tmp_path):
    node = f"test-node-{uuid.uuid4().hex[:6]}"
    lst = Listener(topic_heartbeat(node))
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ML / 'src')!r})
        from vision.events.outbox import Outbox
        from vision.events.publisher import Publisher
        p = Publisher({node!r}, "run-kill", Outbox(":memory:"), heartbeat_s=0.2).start()
        time.sleep(30)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        assert wait(lambda: any(parse("heartbeat", p).status == "online" for _, p in lst.msgs), timeout=10)
        proc.kill()                                             # «упал» без DISCONNECT
        assert wait(lambda: any(parse("heartbeat", p).status == "offline" for _, p in lst.msgs), timeout=20)
    finally:
        proc.kill()
        lst.close()
    will = parse("heartbeat", next(p for _, p in lst.msgs if parse("heartbeat", p).status == "offline"))
    assert will.run_id == "run-kill" and will.sent_at is None
