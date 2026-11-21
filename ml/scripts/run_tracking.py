"""Шаги 2–3: несколько камер -> ОДИН батч YOLO -> свой ByteTrack и свои правила на каждую камеру.

    python scripts/run_tracking.py --camera cam-01                          # с правилами и зонами из конфига
    python scripts/run_tracking.py --camera cam-01 --camera cam-rtsp
    python scripts/run_tracking.py --source data/demo1.mp4 --copies 4      # 4 «камеры» из одного ролика
    python scripts/run_tracking.py --source data/demo1.mp4 --copies 4 --fast --no-show
    python scripts/run_tracking.py --source data/adl-01_rgb.avi --rules-from cam-fall  # чужой ролик, правила cam-fall

Окно: сетка камер, рамки с track_id, зоны (красные). q / Esc — выход.
События правил печатаются в лог (в BE по MQTT их отправит шаг 8).
--fast отключает loop файлов, чтобы прогон закончился.
В конце печатает статистику по трекам каждой камеры (основа для подсчёта смен ID).
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

from vision.analytics.tracker import CameraTracker  # noqa: E402
from vision.analytics.zones import foot_point, load_zones  # noqa: E402
from vision.rules import RuleContext, build_rules, required_inputs  # noqa: E402
from vision.inference.detector import BatchDetector  # noqa: E402
from vision.inference.pose import PoseEstimator, attach_poses  # noqa: E402
from vision.runtime.bus import FrameBus  # noqa: E402
from vision.sources.video_source import VideoSource  # noqa: E402


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8")) or {}


def color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def draw(img: np.ndarray, cam_id: str, ts: str, tracks, zones=(), alert_ids=(),
         poses=None, phases=None) -> np.ndarray:
    img = img.copy()
    h, w = img.shape[:2]
    for z in zones:
        pts = np.array([(x * w, y * h) for x, y in z.polygon], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
        cv2.polylines(img, [pts], True, (0, 0, 255), 3)
        cv2.putText(img, z.zone_id, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
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
        if phase and phase != "upright":
            cv2.putText(img, phase, (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        fx, fy = foot_point(t.bbox, w, h)
        cv2.circle(img, (int(fx * w), int(fy * h)), 6, (0, 0, 255) if t.track_id in alert_ids else c, -1)
    cv2.putText(img, f"{cam_id}  {ts}  persons: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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


def log_event(d) -> None:
    """Пока нет MQTT (шаг 8) — событие в лог одной строкой JSON."""
    msg = {
        "incident_type": d.incident_type, "event_type": d.event_type, "camera_id": d.camera_id,
        "occurred_at": d.occurred_at.isoformat(timespec="milliseconds"),
        "confidence": d.confidence, "zone_id": d.zone_id, "zone_version": d.zone_version,
        "participants": [{"track_id": p.track_id, "bbox": [round(v, 3) for v in p.bbox],
                          "confidence": p.confidence} for p in d.participants],
    }
    logging.warning("EVENT %s", json.dumps(msg, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", action="append", default=[], help="id из cameras.yaml (можно несколько)")
    p.add_argument("--source", help="файл или rtsp:// вместо cameras.yaml")
    p.add_argument("--copies", type=int, default=1, help="сколько виртуальных камер из --source")
    p.add_argument("--rules-from", help="для --source: взять правила у этой камеры из cameras.yaml")
    p.add_argument("--fps", type=float, default=None, help="process_fps (по умолчанию из конфига / 8)")
    p.add_argument("--fast", action="store_true", help="файлы без ожидания (бенчмарк)")
    p.add_argument("--no-show", action="store_true", help="без окна, только статистика")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    app = load_yaml("app.yaml")
    if args.source:
        rules_cfg = {}
        if args.rules_from:
            known = {c["id"]: c for c in load_yaml("cameras.yaml").get("cameras", [])}
            if args.rules_from not in known:
                p.error(f"нет в cameras.yaml: {args.rules_from}")
            rules_cfg = {k: v for k, v in (known[args.rules_from].get("rules") or {}).items()
                         if k != "zone_intrusion"}            # зоны привязаны к своей камере
        cams = [{"id": f"cam-{i + 1:02d}", "source": args.source, "rules": rules_cfg}
                for i in range(args.copies)]
    elif args.camera:
        known = {c["id"]: c for c in load_yaml("cameras.yaml").get("cameras", [])}
        missing = [c for c in args.camera if c not in known]
        if missing:
            p.error(f"нет в cameras.yaml: {missing}")
        cams = [known[c] for c in args.camera]
    else:
        p.error("укажи --camera или --source")

    zones = load_zones(ROOT / "configs" / "zones.json")
    sources, trackers, rules = [], {}, {}
    for cam in cams:
        fps = args.fps or cam.get("process_fps", 8)
        src_path = str(cam["source"])
        if not src_path.startswith(("rtsp", "http")) and not Path(src_path).is_absolute():
            src_path = str(ROOT / src_path)
        sources.append(VideoSource(cam["id"], src_path, process_fps=fps,
                                   realtime=not args.fast and cam.get("realtime", True),
                                   loop=cam.get("loop", False) and not args.fast))
        trackers[cam["id"]] = CameraTracker.from_config(cam["id"], fps, app["tracker"])
        rules[cam["id"]] = build_rules(cam["id"], cam.get("rules"), zones)
        logging.info("camera %s: правила %s", cam["id"], [type(r).__name__ for r in rules[cam["id"]]])

    det_cfg = dict(app["detector"])
    det_cfg["weights"] = str(ROOT / det_cfg["weights"])
    detector = BatchDetector.from_config(det_cfg)
    detector.warmup(len(cams))

    pose_cams = {cid for cid, rs in rules.items() if "poses" in required_inputs(rs)}
    pose_model, match_iou = None, 0.5
    if pose_cams:
        pose_cfg = dict(app["pose"])
        pose_cfg.pop("mode", None)
        match_iou = pose_cfg.pop("match_iou", 0.5)
        pose_cfg["weights"] = str(ROOT / pose_cfg["weights"])
        pose_model = PoseEstimator.from_config(pose_cfg)
        pose_model.warmup(len(pose_cams))
        logging.info("pose-модель для камер: %s", sorted(pose_cams))

    # статистика: track_id -> число кадров, в которых он был
    seen: dict[str, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    n_frames = defaultdict(int)
    batch_sizes, infer_ms, pose_ms = [], [], []
    last_tile: dict[str, np.ndarray] = {}
    sessions: dict[str, int] = {}
    alerts: dict[str, dict[int, float]] = defaultdict(dict)   # track_id -> до какого момента подсвечивать
    n_events = defaultdict(int)

    bus = FrameBus(sources, drop_old=not args.fast).start()
    t_start = time.monotonic()
    try:
        while not bus.done:
            batch = bus.next_batch(timeout=1.0)
            if not batch:
                continue
            t0 = time.perf_counter()
            dets = detector([f.image for f in batch])          # ОДИН прогон на все камеры
            infer_ms.append((time.perf_counter() - t0) * 1000)
            batch_sizes.append(len(batch))

            # поза — вторым батчем, только кадры камер, которым она нужна
            need = [i for i, f in enumerate(batch) if f.camera_id in pose_cams]
            pose_by_idx = {}
            if need:
                t0 = time.perf_counter()
                for i, ps in zip(need, pose_model([batch[i].image for i in need])):
                    pose_by_idx[i] = ps
                pose_ms.append((time.perf_counter() - t0) * 1000)

            for idx, (f, d) in enumerate(zip(batch, dets)):                       # индекс в батче -> камера -> её трекер
                tracks = trackers[f.camera_id].update(d, f.image, session=f.session)
                n_frames[f.camera_id] += 1
                for t in tracks:
                    seen[f.camera_id][(f.session, t.track_id)] += 1

                cam_rules = rules[f.camera_id]
                if sessions.get(f.camera_id) not in (None, f.session):
                    for r in cam_rules:
                        r.reset()                               # после переподключения старые id недействительны
                sessions[f.camera_id] = f.session
                h, w = f.image.shape[:2]
                poses = attach_poses(tracks, pose_by_idx[idx], match_iou) if idx in pose_by_idx else None
                ctx = RuleContext(f.camera_id, f.ts, w, h, tracks, session=f.session, poses=poses)
                for r in cam_rules:
                    for d in r.update(ctx):
                        n_events[f.camera_id] += 1
                        log_event(d)
                        for p_ in d.participants[:1]:
                            alerts[f.camera_id][p_.track_id] = time.monotonic() + 2

                if not args.no_show:
                    now = time.monotonic()
                    hot = {tid for tid, until in alerts[f.camera_id].items() if until > now}
                    cam_zones = [z for r in cam_rules for z in getattr(r, "zones", {}).values()]
                    phases = {}
                    for r in cam_rules:
                        if hasattr(r, "phase_of"):
                            phases.update({t.track_id: r.phase_of(t.track_id) for t in tracks})
                    last_tile[f.camera_id] = draw(f.image, f.camera_id, f.ts_iso, tracks, cam_zones, hot,
                                                  poses, phases)

            if not args.no_show and last_tile:
                cv2.imshow("tracking", grid([last_tile[c["id"]] for c in cams if c["id"] in last_tile]))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        bus.stop()
        cv2.destroyAllWindows()

    wall = time.monotonic() - t_start
    total = sum(n_frames.values())
    print("\n=== итог ===")
    if infer_ms:
        print(f"кадров всего: {total} за {wall:.1f} с ({total / wall:.1f} кадр/с), "
              f"средний батч {np.mean(batch_sizes):.2f}, YOLO на батч {np.mean(infer_ms):.1f} мс"
              + (f", поза на батч {np.mean(pose_ms):.1f} мс" if pose_ms else ""))
    for cam in cams:
        cid = cam["id"]
        lens = sorted(seen[cid].values(), reverse=True)
        short = sum(1 for n in lens if n < 8)
        print(f"{cid}: кадров {n_frames[cid]}, треков {len(lens)}, "
              f"коротких (<8 кадров) {short}, самые длинные: {lens[:5]}, событий {n_events[cid]}")
    print("Смены ID считаются вручную: сколько раз один и тот же человек получил новый id "
          "(много коротких треков — признак частых смен).")


if __name__ == "__main__":
    main()
