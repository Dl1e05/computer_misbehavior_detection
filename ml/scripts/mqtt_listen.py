"""Слушатель MQTT вместо BE: печатает всё, что ML публикует, и проверяет по схеме.

    python scripts/mqtt_listen.py                     # localhost:1883, все топики videoanalytics/v1/#
    python scripts/mqtt_listen.py --host 10.0.0.5 --raw

Каждое сообщение: время приёма, топик, тип, OK / INVALID (+ причина) и краткое содержимое.
--raw — полный JSON. Вектор лица сокращается. Ctrl+C — выход и сводка.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import paho.mqtt.client as mqtt  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from vision.events.schema import TOPIC_PREFIX, message_kind, parse  # noqa: E402


def short(kind: str, d: dict) -> str:
    if kind == "event":
        ps = ",".join(str(p["track_id"]) for p in d.get("participants", []))
        extra = d.get("zone_id") or d.get("episode_id") or (d.get("bbox") and f"bbox={d['bbox']}") or ""
        return (f"{d['incident_type']}/{d['event_type']} cam={d['camera_id']} conf={d['confidence']} "
                f"tracks=[{ps}] {extra} id={d['event_id'][:8]}")
    if kind == "participant":
        face = "лицо" if d.get("embedding") else d.get("snapshot_quality")
        q = d.get("snapshot_quality")
        return f"event={d['event_id'][:8]} track={d['track_id']} {face} q={q} ref={d.get('snapshot_ref')}"
    if kind == "heartbeat":
        cams = ", ".join(f"{c['camera_id']}:{c['status']}" for c in d.get("cameras", []))
        return f"{d['node_id']} {d['status']} run={d['run_id']} [{cams}]"
    return json.dumps(d, ensure_ascii=False)[:200]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--topic", default=f"{TOPIC_PREFIX}/#")
    p.add_argument("--raw", action="store_true", help="печатать полный JSON")
    p.add_argument("--no-heartbeat", action="store_true", help="не печатать heartbeat online")
    args = p.parse_args()
    stats: Counter = Counter()

    def on_connect(client, userdata, flags, rc, props=None):
        print(f"подключено к {args.host}:{args.port}, слушаю {args.topic}", flush=True)
        client.subscribe(args.topic, qos=1)

    def on_message(client, userdata, msg):
        kind = message_kind(msg.topic) or "?"
        try:
            d = json.loads(msg.payload)
        except ValueError:
            stats[f"{kind}:not_json"] += 1
            print(f"{datetime.now():%H:%M:%S.%f}"[:-3], msg.topic, "НЕ JSON", msg.payload[:100], flush=True)
            return
        verdict = "OK"
        if kind != "?":
            try:
                parse(kind, msg.payload)
            except ValidationError as e:
                verdict = "INVALID: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}"
                                                  for x in e.errors()[:3])
        stats[f"{kind}:{'ok' if verdict == 'OK' else 'invalid'}"] += 1
        if args.no_heartbeat and kind == "heartbeat" and d.get("status") == "online" and verdict == "OK":
            return
        if isinstance(d.get("embedding"), list):
            d["embedding"] = f"[{len(d['embedding'])} чисел]"
        body = json.dumps(d, ensure_ascii=False) if args.raw else short(kind, d)
        print(f"{datetime.now():%H:%M:%S.%f}"[:-3], f"{kind:11}", verdict, "|", body, flush=True)

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"mqtt-listen-{datetime.now():%H%M%S}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nитого:", dict(stats))


if __name__ == "__main__":
    main()
