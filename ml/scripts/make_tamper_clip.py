"""Проверочный ролик для правила camera_tampering: обычное видео + вставки саботажа.

    python scripts/make_tamper_clip.py                          # data/demo2.mp4 -> data/tamper_test.mp4
    python scripts/run_tracking.py --camera cam-tamper --fast --no-show

Отрезки (секунды) печатаются и пишутся рядом в .csv — это разметка для проверки:
  covered (тёмная однородная картинка — рука), defocused (сильное размытие),
  glare (засветка фонариком), moved (камеру развернули: кадр сдвинут), между ними — норма.
Настоящие постановочные дубли C1–C5 — в docs/shooting_plan.md.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = [("normal", 8), ("covered", 5), ("normal", 6), ("defocused", 5), ("normal", 6), ("glare", 5),
        ("normal", 6), ("moved", 8), ("normal", 6)]


def tamper(frame: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "covered":
        noise = rng.normal(0, 3, frame.shape)
        return np.clip(np.full(frame.shape, 28, np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "defocused":
        return cv2.GaussianBlur(frame, (0, 0), 10)
    if kind == "glare":
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), np.float32)
        cv2.circle(mask, (w // 2, h // 2), int(0.6 * max(h, w)), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 40)[..., None]
        return np.clip(frame * (1 - mask) + 255 * mask * 1.3, 0, 255).astype(np.uint8)
    if kind == "moved":
        h, w = frame.shape[:2]
        return np.roll(frame, (h // 4, w // 3), (0, 1))
    return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(ROOT / "data/demo2.mp4"))
    p.add_argument("--out", default=str(ROOT / "data/tamper_test.mp4"))
    args = p.parse_args()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, 960 / w)
    size = (int(w * scale), int(h * scale))
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    t, rows = 0.0, []
    for kind, dur in PLAN:
        for _ in range(int(dur * fps)):
            ok, frame = cap.read()
            if not ok:                                         # ролик короче — по кругу
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            out.write(tamper(cv2.resize(frame, size), kind, rng))
        if kind != "normal":
            rows.append((kind, t, t + dur))
        t += dur
    out.release()
    with Path(args.out).with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tamper_type", "start_s", "end_s"])
        wr.writerows(rows)
    print(f"-> {args.out} ({t:.0f} с): " + ", ".join(f"{k} {a:.0f}–{b:.0f} с" for k, a, b in rows))


if __name__ == "__main__":
    main()
