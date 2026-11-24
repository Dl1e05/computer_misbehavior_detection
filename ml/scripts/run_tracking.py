"""Демо узла: несколько камер -> ОДИН батч YOLO -> свой ByteTrack и правила на камеру -> MQTT.

    python scripts/run_tracking.py --camera cam-01                          # с правилами и зонами из конфига
    python scripts/run_tracking.py --camera cam-01 --camera cam-rtsp
    python scripts/run_tracking.py --source data/demo1.mp4 --copies 4      # 4 «камеры» из одного ролика
    python scripts/run_tracking.py --source data/demo1.mp4 --copies 4 --fast --no-show
    python scripts/run_tracking.py --source data/adl-01_rgb.avi --rules-from cam-fall  # чужой ролик, правила cam-fall

Сам конвейер — vision.runtime.pipeline.Pipeline (тот же гоняет scripts/load_test.py).
Окно: сетка камер, рамки с track_id, зоны (красные). q / Esc — выход.
События и участники — в лог и в MQTT (--no-mqtt — только лог).
--fast: файлы без ожидания и без loop, чтобы прогон закончился.
В конце — статистика по трекам каждой камеры (основа для подсчёта смен ID).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.analytics.zones import foot_point, label, load_zones  # noqa: E402
from vision.runtime.bus import FrameBus  # noqa: E402
from vision.runtime.pipeline import Pipeline  # noqa: E402


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8")) or {}


def color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def draw(img: np.ndarray, cam_id: str, ts: str, tracks, zones=(), alert_ids=(),
         poses=None, phases=None, lines=(), note: str = "", objects=(), object_labels=None) -> np.ndarray:
    img = img.copy()
    h, w = img.shape[:2]
    for o in objects:                                  # предметы: оранжевые, с состоянием (владелец / без присмотра)
        x1, y1, x2, y2 = map(int, o.bbox)
        text = (object_labels or {}).get(o.track_id, "")
        bad = text.startswith(("ABANDONED", "unattended"))
        c = (0, 0, 255) if bad else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, text, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    for ln in lines:                                   # линия: p1 -> p2, стрелка показывает «in» (слева направо)
        (x1, y1), (x2, y2) = [(int(x * w), int(y * h)) for x, y in ln.points]
        cv2.line(img, (x1, y1), (x2, y2), (255, 200, 0), 3)
        mx, my, dx, dy = (x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1
        n = max((dx * dx + dy * dy) ** 0.5, 1)
        cv2.arrowedLine(img, (mx, my), (int(mx - dy / n * 40), int(my + dx / n * 40)), (255, 200, 0), 2)
        cv2.putText(img, label(ln), (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    for z in zones:
        pts = np.array([(x * w, y * h) for x, y in z.polygon], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
        cv2.polylines(img, [pts], True, (0, 0, 255), 3)
        cv2.putText(img, label(z), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        c = color(t.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"id {t.track_id}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        k = (poses or {}).get(t.track_id)
        if k is not None:
            for a, b in SKELETON:
                if k[a, 2] > 0.5 and k[b, 2] > 0.5:
                    cv2.line(img, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), c, 2)
        phase = (phases or {}).get(t.track_id)
        if phase and phase not in ("upright",):
            cv2.putText(img, phase, (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        fx, fy = foot_point(t.bbox, w, h)
        cv2.circle(img, (int(fx * w), int(fy * h)), 6, (0, 0, 255) if t.track_id in alert_ids else c, -1)
    cv2.putText(img, f"{cam_id}  {ts}  persons: {len(tracks)}  {note}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if not note.startswith("TAMPER") else (0, 0, 255), 2)
    return img


def grid(tiles: list[np.ndarray], tile_w: int = 640) -> np.ndarray:
    tiles = [cv2.resize(t, (tile_w, int(t.shape[0] * tile_w / t.shape[1]))) for t in tiles]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cols = 2 if len(tiles) > 1 else 1
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def log_event(e) -> None:
    """Event (контракт шага 8) одной строкой — то же, что уходит в MQTT."""
    logging.warning("EVENT %s", e.to_bytes().decode("utf-8"))


def log_participant(p, reason: str | None = None) -> None:
    """Сообщение участника (топик .../participants); вектор в логе сокращён."""
    msg = json.loads(p.to_bytes())
    if msg.get("embedding"):
        msg["embedding"] = f"[{len(msg['embedding'])} чисел: {msg['embedding'][:3]}...]"
    logging.warning("PARTICIPANT %s%s", json.dumps(msg, ensure_ascii=False),
                    f"  (причина: {reason})" if reason else "")


def build_cams(args, p) -> list[dict]:
    if args.source:
        rules_cfg = {}
        if args.rules_from:
            known = {c["id"]: c for c in load_yaml("cameras.yaml").get("cameras", [])}
            if args.rules_from not in known:
                p.error(f"нет в cameras.yaml: {args.rules_from}")
            rules_cfg = {k: v for k, v in (known[args.rules_from].get("rules") or {}).items()
                         if k != "zone_intrusion"}            # зоны привязаны к своей камере
        return [{"id": f"cam-{i + 1:02d}", "source": args.source, "rules": rules_cfg}
                for i in range(args.copies)]
    if args.camera:
        known = {c["id"]: c for c in load_yaml("cameras.yaml").get("cameras", [])}
        missing = [c for c in args.camera if c not in known]
        if missing:
            p.error(f"нет в cameras.yaml: {missing}")
        return [known[c] for c in args.camera]
    p.error("укажи --camera или --source")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", action="append", default=[], help="id из cameras.yaml (можно несколько)")
    p.add_argument("--source", help="файл или rtsp:// вместо cameras.yaml")
    p.add_argument("--copies", type=int, default=1, help="сколько виртуальных камер из --source")
    p.add_argument("--rules-from", help="для --source: взять правила у этой камеры из cameras.yaml")
    p.add_argument("--fps", type=float, default=None, help="process_fps (по умолчанию из конфига / 8)")
    p.add_argument("--fast", action="store_true", help="файлы без ожидания (бенчмарк)")
    p.add_argument("--no-show", action="store_true", help="без окна, только статистика")
    p.add_argument("--no-faces", action="store_true", help="не считать лица участников (шаг 7)")
    p.add_argument("--no-mqtt", action="store_true", help="не отправлять в MQTT, только лог (шаг 8)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cams = build_cams(args, p)
    node = Pipeline(load_yaml("app.yaml"), cams, load_zones(ROOT / "configs" / "zones.json"), ROOT,
                    fast=args.fast, fps=args.fps, faces=not args.no_faces, mqtt=not args.no_mqtt,
                    on_event=log_event, on_participant=log_participant)
    logging.info("статус узла:\n%s", node.status.describe())

    seen: dict[str, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    last_tile: dict[str, np.ndarray] = {}
    alerts: dict[str, dict[int, float]] = defaultdict(dict)   # track_id -> до какого момента подсвечивать

    bus = FrameBus(node.sources, drop_old=not args.fast).start()
    t_start = time.monotonic()
    try:
        while not bus.done:
            outs = node.process(bus.next_batch(timeout=1.0))
            for o in outs:
                cid = o.frame.camera_id
                for t in o.tracks:
                    seen[cid][(o.frame.session, t.track_id)] += 1
                for e in o.events:
                    for p_ in e.participants[:1]:
                        alerts[cid][p_.track_id] = time.monotonic() + 2
                if not args.no_show:
                    now = time.monotonic()
                    hot = {tid for tid, until in alerts[cid].items() if until > now}
                    cam_zones = [z for r in o.rules for z in getattr(r, "zones", {}).values()]
                    phases = {}
                    for r in o.rules:
                        if hasattr(r, "phase_of"):
                            phases.update({t.track_id: r.phase_of(t.track_id) for t in o.tracks})
                    cam_lines = [ln for r in o.rules for ln in getattr(r, "lines", {}).values()]
                    tamper = next((r.state for r in o.rules if r.incident_type == "camera_tampering"), None)
                    note = f"TAMPER: {tamper}" if tamper not in (None, "normal") else ""
                    obj_rule = next((r for r in o.rules if hasattr(r, "state_of")), None)
                    obj_labels = {t.track_id: f"{node.object_names.get(t.cls, t.cls)} {obj_rule.state_of(t.track_id) or ''}"
                                  for t in o.objects} if obj_rule else {}
                    last_tile[cid] = draw(o.frame.image, cid, o.frame.ts_iso, o.tracks, cam_zones, hot,
                                          o.poses, phases, cam_lines, note, o.objects, obj_labels)
            if not args.no_show and last_tile:
                cv2.imshow("tracking", grid([last_tile[c["id"]] for c in cams if c["id"] in last_tile]))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        bus.stop()
        node.close()
        if node.publisher:
            left = len(node.publisher.outbox)
            print(f"MQTT: подтверждено брокером {node.publisher.sent}, heartbeat {node.publisher.heartbeats}, "
                  f"осталось в outbox {left}" + (" (уйдут при следующем запуске)" if left else ""))
        cv2.destroyAllWindows()

    m = node.metrics
    wall = time.monotonic() - t_start
    total = sum(m.frames.values())
    print("\n=== итог ===")
    if m.det_ms:
        print(f"кадров всего: {total} за {wall:.1f} с ({total / wall:.1f} кадр/с), "
              f"средний батч {np.mean(m.batch_size):.2f}, YOLO на батч {np.mean(m.det_ms):.1f} мс"
              + (f", поза на батч {np.mean(m.pose_ms):.1f} мс" if m.pose_ms else "")
              + (f", огонь на батч {np.mean(m.fire_ms):.1f} мс" if m.fire_ms else ""))
    print("статус узла:\n" + node.status.describe())
    for cam in cams:
        cid = cam["id"]
        lens = sorted(seen[cid].values(), reverse=True)
        short = sum(1 for n in lens if n < 8)
        print(f"{cid}: кадров {m.frames[cid]}, треков {len(lens)}, "
              f"коротких (<8 кадров) {short}, самые длинные: {lens[:5]}, событий {m.events[cid]}"
              + (f", участников: {dict(m.participants[cid])}" if cid in node.collectors else ""))
    print("Смены ID считаются вручную: сколько раз один и тот же человек получил новый id "
          "(много коротких треков — признак частых смен).")


if __name__ == "__main__":
    main()
