"""Нарисовать зону (или линию) мышкой на кадре камеры и сохранить в configs/zones.json.

Пока зоны не приходят из веба (API BE), рисуем их сами.

    python scripts/draw_zone.py --camera cam-01 --zone zone-1
    python scripts/draw_zone.py --camera cam-01 --zone zone-1 --at 5      # кадр на 5-й секунде
    python scripts/draw_zone.py --source data/x.mp4 --camera-id cam-05 --zone zone-7
    python scripts/draw_zone.py --camera cam-01 --zone line-door-1 --line  # линия: 2 точки, p1 -> p2
                                     # «in» = переход слева направо, если смотреть от p1 к p2
Расписание зоны («после 22:00») пока задаётся руками в zones.json: поле "schedule" (см. analytics/zones.py).

Управление: ЛКМ — точка, ПКМ или Backspace — убрать последнюю, C — очистить,
Enter — сохранить, Esc — выйти без сохранения.
Координаты пишутся в долях 0–1 от кадра. Перерисовал существующую зону — её
version увеличится на 1 (правило по версии поймёт, что зона изменилась).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.analytics.zones import Line, Zone, load_zones, save_zones  # noqa: E402

MAX_W = 1280   # окно не шире этого, координаты всё равно считаются от исходного кадра


def grab_frame(source: str, at_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"не открыть источник: {source}")
    if at_s > 0 and not source.lower().startswith(("rtsp", "http")):
        cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("не удалось прочитать кадр")
    return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", help="id из configs/cameras.yaml (источник возьмётся оттуда)")
    p.add_argument("--source", help="файл или rtsp:// (тогда нужен --camera-id)")
    p.add_argument("--camera-id", help="к какой камере относится зона при --source")
    p.add_argument("--zone", required=True, help="id зоны или линии, например zone-1 / line-door-1")
    p.add_argument("--line", action="store_true", help="рисовать линию (2 точки) вместо зоны")
    p.add_argument("--at", type=float, default=0.0, help="секунда ролика для кадра")
    p.add_argument("--out", default=str(ROOT / "configs" / "zones.json"))
    args = p.parse_args()

    if args.camera:
        cams = yaml.safe_load((ROOT / "configs" / "cameras.yaml").read_text(encoding="utf-8"))
        cam = next((c for c in cams.get("cameras", []) if c["id"] == args.camera), None)
        if cam is None:
            p.error(f"нет камеры {args.camera} в cameras.yaml")
        camera_id, source = cam["id"], os.path.expandvars(str(cam["source"]))
    elif args.source and args.camera_id:
        camera_id, source = args.camera_id, args.source
    else:
        p.error("укажи --camera или --source вместе с --camera-id")
    if not source.lower().startswith(("rtsp", "http")) and not Path(source).is_absolute():
        source = str(ROOT / source)

    frame = grab_frame(source, args.at)
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_W / w)
    base = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame.copy()

    zones = load_zones(args.out)
    old = zones.get(args.zone)
    if old is not None and isinstance(old, Line) != args.line:
        p.error(f"{args.zone} уже есть как {'линия' if isinstance(old, Line) else 'зона'}")
    points: list[tuple[float, float]] = (list(old.points if isinstance(old, Line) else old.polygon)
                                         if old and old.camera_id == camera_id else [])

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            if args.line and len(points) >= 2:
                return
            points.append((round(x / base.shape[1], 4), round(y / base.shape[0], 4)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    win = f"zone {args.zone} @ {camera_id}"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = base.copy()
        # остальные зоны этой камеры — серым, для ориентира
        for key, z in zones.items():
            if z.camera_id == camera_id and key != args.zone:
                poly = z.points if isinstance(z, Line) else z.polygon
                pts = np.array([(x * img.shape[1], y * img.shape[0]) for x, y in poly], np.int32)
                cv2.polylines(img, [pts], not isinstance(z, Line), (160, 160, 160), 1)
        if points:
            pts = np.array([(x * img.shape[1], y * img.shape[0]) for x, y in points], np.int32)
            overlay = img.copy()
            if len(points) >= 3 and not args.line:
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            cv2.polylines(img, [pts], len(points) >= 3, (0, 0, 255), 2)
            for pt in pts:
                cv2.circle(img, tuple(int(v) for v in pt), 4, (0, 255, 255), -1)
        cv2.putText(img, "LMB: point  RMB/Backspace: undo  C: clear  Enter: save  Esc: exit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            print("выход без сохранения")
            break
        if key == 8 and points:
            points.pop()
        elif key in (ord("c"), ord("C")):
            points.clear()
        elif key in (13, 10):
            try:
                version = old.version + 1 if old else 1
                if args.line:
                    zones[args.zone] = Line(args.zone, camera_id, points, version=version)
                else:
                    schedule = old.schedule if isinstance(old, Zone) else None   # расписание сохраняем
                    zones[args.zone] = Zone(args.zone, camera_id, points, version=version, schedule=schedule)
            except ValueError as e:
                print("не сохранено:", e)
                continue
            save_zones(args.out, zones)
            print(f"сохранено: {args.zone} (камера {camera_id}, {len(points)} точек, "
                  f"version {version}) -> {args.out}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
