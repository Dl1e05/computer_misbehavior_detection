"""Шаг 2.1: «пощупать» встроенный model.track на одном ролике (только для знакомства).

    python scripts/track_demo.py                 # data/demo1.mp4
    python scripts/track_demo.py --source data/other.mp4
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--source", default=str(ROOT / "data" / "demo1.mp4"))
p.add_argument("--weights", default=str(ROOT / "weights" / "yolo26n.pt"))
args = p.parse_args()

model = YOLO(args.weights)
# трекер указываем явно: по умолчанию в свежих ultralytics стоит не ByteTrack
for _ in model.track(args.source, stream=True, tracker="bytetrack.yaml",
                     classes=[0], show=True, verbose=False):
    pass
