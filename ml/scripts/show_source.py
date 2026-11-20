"""Проверка шага 1: показывает видео камеры с временем кадра.

Примеры:
  python scripts/show_source.py --camera cam-01
  python scripts/show_source.py --source data/demo1.mp4 --fps 8
  python scripts/show_source.py --source rtsp://localhost:8554/cam1

Окно закрывается клавишей q или Esc.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.sources.video_source import VideoSource  # noqa: E402


def camera_from_config(camera_id: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "cameras.yaml").read_text(encoding="utf-8"))
    for cam in cfg.get("cameras", []):
        if cam["id"] == camera_id:
            return cam
    raise SystemExit(f"Камера {camera_id} не найдена в configs/cameras.yaml")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", help="id камеры из configs/cameras.yaml")
    p.add_argument("--source", help="путь к файлу или rtsp://...")
    p.add_argument("--fps", type=float, default=None, help="process_fps")
    p.add_argument("--fast", action="store_true", help="файл без ожидания (бенчмарк)")
    p.add_argument("--loop", action="store_true", help="файл по кругу")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.camera:
        cam = camera_from_config(args.camera)
    elif args.source:
        cam = {"id": "cam-test", "source": args.source}
    else:
        p.error("укажи --camera или --source")

    src = VideoSource(
        camera_id=cam["id"],
        source=str(cam["source"]),
        process_fps=args.fps or cam.get("process_fps", 8),
        realtime=not args.fast and cam.get("realtime", True),
        loop=args.loop or cam.get("loop", False),
    )

    try:
        for f in src:
            img = f.image.copy()
            text = f"{f.camera_id}  {f.ts_iso}  #{f.index}  session {f.session}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f.camera_id, img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()