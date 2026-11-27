"""M1: обучение YOLO26n огонь/дым на D-Fire (Фаза 3.1).

    python training/fire/train.py --name smoke-check --epochs 1 --fraction 0.05   # проверка, что всё запускается
    python training/fire/train.py --name v1                                       # полное обучение

Каждый запуск — строка в reports/experiments.csv (id, дата, данные, параметры, метрики val, веса).
test здесь НЕ запускается: только один раз в конце, отдельно (ТЗ 5.2).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training" / "fire" / "data.yaml"
EXPERIMENTS = ROOT / "reports" / "experiments.csv"
FIELDS = ["id", "date", "model", "data", "data_hash", "params", "best_epoch", "val_mAP50", "val_mAP50-95",
          "val_P", "val_R", "per_class", "weights", "note"]


def resolved_data() -> str:
    """Ultralytics считает относительный path от своей папки datasets, а не от yaml — даём абсолютный."""
    import yaml

    cfg = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    cfg["path"] = str((ROOT / cfg["path"]).resolve())
    out = ROOT / cfg["path"] / "data.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)


def lists_hash() -> str:
    h = hashlib.sha256()
    for split in ("train", "val"):
        h.update((ROOT / "data" / "splits" / f"fire_img_{split}_v1.txt").read_bytes())
    return h.hexdigest()[:12]


def log_experiment(row: dict) -> None:
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    new = not EXPERIMENTS.exists()
    with EXPERIMENTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=str(ROOT / "weights" / "yolo26n.pt"))   # старт с весов COCO
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=float, default=-1)      # -1: под ~60% видеопамяти
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=1.0)  # доля train — для быстрой проверки
    ap.add_argument("--workers", type=int, default=2)   # 8 на ноутбуке (16 ГБ) -> ошибка 1455, нет памяти
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = resolved_data()
    run_dir = ROOT / "runs" / "fire" / args.name
    if args.resume:
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
        model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=batch,
                    patience=args.patience, fraction=args.fraction, workers=args.workers,
                    seed=0, deterministic=True, project=str(ROOT / "runs" / "fire"), name=args.name,
                    exist_ok=True, plots=True)

    best = run_dir / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=data, split="val", imgsz=args.imgsz, plots=False, verbose=False)
    names = m.names
    per_class = {names[c]: {"P": round(float(m.box.p[i]), 3), "R": round(float(m.box.r[i]), 3),
                            "mAP50": round(float(m.box.ap50[i]), 3)}
                 for i, c in enumerate(m.box.ap_class_index)}
    results = run_dir / "results.csv"
    best_epoch = ""
    if results.exists():
        rows = list(csv.DictReader(results.open(encoding="utf-8")))
        key = next((k for k in rows[0] if "mAP50-95" in k), None) if rows else None
        if key:
            best_epoch = max(rows, key=lambda r: float(r[key]))["epoch"].strip()
    log_experiment({
        "id": f"fire-{args.name}", "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": Path(args.model).name, "data": "d-fire v1 (fire_img_*_v1.txt)", "data_hash": lists_hash(),
        "params": json.dumps({k: getattr(args, k) for k in ("epochs", "imgsz", "batch", "patience", "fraction")}),
        "best_epoch": best_epoch, "val_mAP50": round(float(m.box.map50), 4),
        "val_mAP50-95": round(float(m.box.map), 4), "val_P": round(float(m.box.mp), 4),
        "val_R": round(float(m.box.mr), 4), "per_class": json.dumps(per_class, ensure_ascii=False),
        "weights": str(best.relative_to(ROOT)), "note": args.note,
    })
    print("val:", json.dumps(per_class, ensure_ascii=False), "| mAP50", round(float(m.box.map50), 4))


if __name__ == "__main__":                    # на Windows без этого Ultralytics падает (dataloader workers)
    main()
