"""Энролмент Базы лиц: фото людей с подтверждённым согласием -> векторы той же версии, что в событиях.

Пока API BE нет, вход — папка, выход — JSON:

    data/faces/enroll/<person_id>/*.jpg      # 3–5 фото на человека, только с согласием (ТЗ 13.4)

    python scripts/enroll_faces.py                                  # data/faces/enroll -> data/faces/enrolled.json
    python scripts/enroll_faces.py --src path/to/photos --out out.json

На каждое фото: ok + embedding (512, норма 1) + quality, или rejected + причина
(no_face / several_faces / too_small / low_det_score / low_quality / unreadable /
not_like_others — фото не похоже на остальные фото этого человека: чужое фото в записи).
Тот же скрипт пересчитывает базу при смене embedding_model_version.
Потом: список фото и запись векторов — через API BE.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.face.face_service import FaceService  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)          # cv2.imread не любит кириллицу в пути
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "data/faces/enroll"), help="папка <person_id>/<фото>")
    p.add_argument("--out", default=str(ROOT / "data/faces/enrolled.json"))
    p.add_argument("--same-min", type=float, default=0.3,
                   help="мин. медианное сходство фото с остальными фото того же человека")
    args = p.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        p.error(f"нет папки {src}: положи фото как {src}/<person_id>/*.jpg")
    photos = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXT)
    if not photos:
        p.error(f"в {src} нет фото")

    cfg = dict(yaml.safe_load((ROOT / "configs/app.yaml").read_text(encoding="utf-8"))["face"])
    cfg["det_size"] = cfg.get("enroll_det_size", 640)        # целые фото, не маленькие кропы
    service = FaceService.from_config(cfg)

    records, stats = [], Counter()
    for f in photos:
        person = f.parent.name if f.parent != src else f.stem
        img = read_image(f)
        rec = {"person_id": person, "file": f.relative_to(src).as_posix(),
               "embedding_model_version": service.model_version}
        if img is None:
            rec.update(status="rejected", reason="unreadable")
        else:
            r = service.enroll(img)
            if r.usable:
                rec.update(status="ok", quality=r.quality,
                           embedding=[round(float(v), 6) for v in r.embedding])
            else:
                rec.update(status="rejected", reason=r.reason)
        records.append(rec)

    # согласованность: все фото записи должны быть одним человеком (нужно >= 3 фото, чтобы понять, какое лишнее)
    by_person: dict[str, list[dict]] = {}
    for rec in records:
        if rec["status"] == "ok":
            by_person.setdefault(rec["person_id"], []).append(rec)
    for person, recs in by_person.items():
        vecs = [np.array(r["embedding"], np.float32) for r in recs]
        for i, rec in enumerate(recs):
            others = [float(vecs[i] @ v) for j, v in enumerate(vecs) if j != i]
            if not others:
                continue
            rec["same_person_sim"] = round(float(np.median(others)), 3)
            if rec["same_person_sim"] < args.same_min:
                if len(recs) >= 3:
                    rec.update(status="rejected", reason="not_like_others")
                    rec.pop("embedding")
                else:
                    rec["warning"] = "2 фото не похожи друг на друга — проверить вручную"

    for rec in records:
        stats[rec["status"] if rec["status"] == "ok" else rec["reason"]] += 1
        info = rec.get("quality", rec.get("reason"))
        extra = f"  сходство со своими {rec['same_person_sim']}" if "same_person_sim" in rec else ""
        print(f"{rec['status']:8} {rec['file']:40} {info}{extra}{'  ! ' + rec['warning'] if 'warning' in rec else ''}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "embedding_model_version": service.model_version,
                               "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    people = {r["person_id"] for r in records if r["status"] == "ok"}
    print(f"\nфото: {len(records)}, итог: {dict(stats)}, людей с вектором: {len(people)} -> {out}")


if __name__ == "__main__":
    main()
