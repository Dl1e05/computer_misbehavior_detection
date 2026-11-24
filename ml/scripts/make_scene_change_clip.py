"""Проверочный ролик для правила scene_change: реальная площадь (demo2, люди ходят) + вставки.

    python scripts/make_scene_change_clip.py                     # data/demo2.mp4 -> data/scene_change_test.mp4
    python scripts/run_tracking.py --camera cam-scene --fast --no-show --no-mqtt

Отрезки:
  0–8 с   норма (правило учит эталон рольставни киоска);
  8–14 с  весь кадр темнее (облако / убавили свет) — события быть НЕ должно;
  14–16 с норма;
  16–30 с на рольставне киоска «граффити» (яркие пятна и полосы) — событие ~через hold_s.
Зона рольставни — kiosk-shutter в configs/zones.json. Разметка — рядом в .csv.
Настоящие постановочные дубли V1–V3 (вандализм по макету) — в docs/shooting_plan.md.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHUTTER = (0.245, 0.19, 0.38, 0.345)            # x1, y1, x2, y2 в долях кадра
PLAN = [("normal", 8), ("darker", 6), ("normal", 2), ("graffiti", 14)]


def graffiti_layer(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Слой «граффити» на рольставне: картинка и маска (одни и те же весь отрезок)."""
    rng = np.random.default_rng(3)
    layer = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = int(SHUTTER[0] * w), int(SHUTTER[1] * h), int(SHUTTER[2] * w), int(SHUTTER[3] * h)
    for _ in range(40):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        p1 = (int(rng.integers(x1, x2)), int(rng.integers(y1, y2)))
        p2 = (int(rng.integers(x1, x2)), int(rng.integers(y1, y2)))
        th = int(rng.integers(6, 18))
        cv2.line(layer, p1, p2, color, th)
        cv2.line(mask, p1, p2, 255, th)
    cv2.rectangle(mask, (0, 0), (w, y1), 0, -1)
    cv2.rectangle(mask, (0, y2), (w, h), 0, -1)
    cv2.rectangle(mask, (0, 0), (x1, h), 0, -1)
    cv2.rectangle(mask, (x2, 0), (w, h), 0, -1)
    return layer, mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(ROOT / "data/demo2.mp4"))
    p.add_argument("--out", default=str(ROOT / "data/scene_change_test.mp4"))
    args = p.parse_args()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, 960 / w)
    size = (int(w * scale), int(h * scale))
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    layer, mask = graffiti_layer(size[1], size[0])
    t, rows = 0.0, []
    for kind, dur in PLAN:
        for _ in range(int(dur * fps)):
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            frame = cv2.resize(frame, size)
            if kind == "darker":
                frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=0)
            elif kind == "graffiti":
                frame[mask > 0] = layer[mask > 0]
            out.write(frame)
        rows.append((kind, t, t + dur))
        t += dur
    out.release()
    with Path(args.out).with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["segment", "start_s", "end_s"])
        wr.writerows(rows)
    print(f"-> {args.out} ({t:.0f} с): " + ", ".join(f"{k} {a:.0f}–{b:.0f} с" for k, a, b in rows))


if __name__ == "__main__":
    main()
