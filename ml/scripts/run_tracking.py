"""Шаг 2.2: несколько камер -> ОДИН батч YOLO -> свой ByteTrack на каждую камеру.

    python scripts/run_tracking.py --camera cam-01 --camera cam-rtsp
    python scripts/run_tracking.py --source data/demo1.mp4 --copies 4      # 4 «камеры» из одного ролика
    python scripts/run_tracking.py --source data/demo1.mp4 --copies 4 --fast --no-show

Окно: сетка камер, рамки с track_id. q / Esc — выход.
В конце печатает статистику по трекам каждой камеры (основа для подсчёта смен ID).
"""
from __future__ import annotations

import argparse
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
from vision.inference.detector import BatchDetector  # noqa: E402
from vision.runtime.bus import FrameBus  # noqa: E402
from vision.sources.video_source import VideoSource  # noqa: E402


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8")) or {}


def color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def draw(img: np.ndarray, cam_id: str, ts: str, tracks) -> np.ndarray:
    img = img.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        c = color(t.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"id {t.track_id}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", action="append", default=[], help="id из cameras.yaml (можно несколько)")
    p.add_argument("--source", help="файл или rtsp:// вместо cameras.yaml")
    p.add_argument("--copies", type=int, default=1, help="сколько виртуальных камер из --source")
    p.add_argument("--fps", type=float, default=None, help="process_fps (по умолчанию из конфига / 8)")
    p.add_argument("--fast", action="store_true", help="файлы без ожидания (бенчмарк)")
    p.add_argument("--no-show", action="store_true", help="без окна, только статистика")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    app = load_yaml("app.yaml")
    if args.source:
        cams = [{"id": f"cam-{i + 1:02d}", "source": args.source} for i in range(args.copies)]
    elif args.camera:
        known = {c["id"]: c for c in load_yaml("cameras.yaml").get("cameras", [])}
        missing = [c for c in args.camera if c not in known]
        if missing:
            p.error(f"нет в cameras.yaml: {missing}")
        cams = [known[c] for c in args.camera]
    else:
        p.error("укажи --camera или --source")

    sources, trackers = [], {}
    for cam in cams:
        fps = args.fps or cam.get("process_fps", 8)
        src_path = str(cam["source"])
        if not src_path.startswith(("rtsp", "http")) and not Path(src_path).is_absolute():
            src_path = str(ROOT / src_path)
        sources.append(VideoSource(cam["id"], src_path, process_fps=fps,
                                   realtime=not args.fast and cam.get("realtime", True),
                                   loop=cam.get("loop", False)))
        trackers[cam["id"]] = CameraTracker.from_config(cam["id"], fps, app["tracker"])

    det_cfg = dict(app["detector"])
    det_cfg["weights"] = str(ROOT / det_cfg["weights"])
    detector = BatchDetector.from_config(det_cfg)
    detector.warmup(len(cams))

    # статистика: track_id -> число кадров, в которых он был
    seen: dict[str, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    n_frames = defaultdict(int)
    batch_sizes, infer_ms = [], []
    last_tile: dict[str, np.ndarray] = {}

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

            for f, d in zip(batch, dets):                       # индекс в батче -> камера -> её трекер
                tracks = trackers[f.camera_id].update(d, f.image, session=f.session)
                n_frames[f.camera_id] += 1
                for t in tracks:
                    seen[f.camera_id][(f.session, t.track_id)] += 1
                if not args.no_show:
                    last_tile[f.camera_id] = draw(f.image, f.camera_id, f.ts_iso, tracks)

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
              f"средний батч {np.mean(batch_sizes):.2f}, YOLO на батч {np.mean(infer_ms):.1f} мс")
    for cam in cams:
        cid = cam["id"]
        lens = sorted(seen[cid].values(), reverse=True)
        short = sum(1 for n in lens if n < 8)
        print(f"{cid}: кадров {n_frames[cid]}, треков {len(lens)}, "
              f"коротких (<8 кадров) {short}, самые длинные: {lens[:5]}")
    print("Смены ID считаются вручную: сколько раз один и тот же человек получил новый id "
          "(много коротких треков — признак частых смен).")


if __name__ == "__main__":
    main()
