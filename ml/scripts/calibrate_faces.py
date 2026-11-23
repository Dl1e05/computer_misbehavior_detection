"""Калибровка порога сходства лиц (Фаза 3.5, задача M6) — на парах фото в формате LFW.

    python scripts/calibrate_faces.py                       # LFW из data/raw/lfw (скачан через scikit-learn)
    python scripts/calibrate_faces.py --images DIR --pairs pairs.txt --name my_volunteers

Что делает:
1. Векторы для всех фото (тот же FaceService и та же версия, что в событиях); кэш в .npz.
2. Верификация по парам «тот же / разные»: 10-fold точность (стандарт LFW), FAR/FRR для
   порогов, EER, пороги при FAR 1% и 0.1%.
3. Опознание как у BE: у людей с >= 4 фото первые 3 — «База лиц», остальные — запросы;
   лучшее сходство с базой -> кандидат или «совпадений нет» при выбранном пороге.
Итог — reports/faces/<name>_calibration.json и .png.

ВАЖНО: порог на LFW (фото знаменитостей, крупно и резко) — только проверка, что сервис
работает. Рабочий порог для BE подбираем на своих добровольцах и своих камерах.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.face.face_service import FaceService  # noqa: E402

LFW = ROOT / "data/raw/lfw/lfw_home"


def parse_pairs(path: Path):
    """pairs.txt LFW: строка 'folds n', затем по фолду n пар «тот же» и n «разные»."""
    lines = path.read_text().split("\n")
    folds, n = map(int, lines[0].split())
    pairs = []
    for i, line in enumerate(l for l in lines[1:] if l.strip()):
        p = line.split("\t")
        fold = i // (2 * n)
        if len(p) == 3:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[0]}/{p[0]}_{int(p[2]):04d}.jpg", 1, fold))
        else:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[2]}/{p[2]}_{int(p[3]):04d}.jpg", 0, fold))
    return pairs, folds


def embed_all(service: FaceService, root: Path, files: list[str], cache: Path):
    emb, quality, reasons = {}, {}, {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        emb, quality, reasons = z["emb"].item(), z["quality"].item(), z["reasons"].item()
    todo = [f for f in files if f not in emb and f not in reasons]
    for i, f in enumerate(todo, 1):
        img = cv2.imdecode(np.fromfile(str(root / f), np.uint8), cv2.IMREAD_COLOR)
        r = service.enroll(img, pick="center") if img is not None else None
        if r is not None and r.usable:
            emb[f], quality[f] = r.embedding, r.quality
        else:
            reasons[f] = r.reason if r is not None else "unreadable"
        if i % 500 == 0 or i == len(todo):
            print(f"  векторы: {i}/{len(todo)}", flush=True)
            np.savez(cache, emb=np.array(emb, dtype=object), quality=np.array(quality, dtype=object),
                     reasons=np.array(reasons, dtype=object))
    return emb, quality, reasons


def far_frr(sims: np.ndarray, same: np.ndarray, thr: float) -> tuple[float, float]:
    far = float(np.mean(sims[~same] >= thr)) if (~same).any() else 0.0    # чужого приняли за своего
    frr = float(np.mean(sims[same] < thr)) if same.any() else 0.0         # своего не узнали
    return far, frr


def best_threshold(sims, same, grid):
    acc = [np.mean((sims >= t) == same) for t in grid]
    return float(grid[int(np.argmax(acc))])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", default=str(LFW / "lfw_funneled"))
    p.add_argument("--pairs", default=str(LFW / "pairs.txt"))
    p.add_argument("--name", default="lfw")
    p.add_argument("--gallery", type=int, default=3, help="фото на человека в «Базе лиц» для опознания")
    args = p.parse_args()

    root = Path(args.images)
    out_dir = ROOT / "reports/faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(yaml.safe_load((ROOT / "configs/app.yaml").read_text(encoding="utf-8"))["face"])
    cfg["det_size"] = cfg.get("enroll_det_size", 640)
    service = FaceService.from_config(cfg)

    pairs, n_folds = parse_pairs(Path(args.pairs))
    all_files = sorted(f.relative_to(root).as_posix() for f in root.rglob("*.jpg"))
    print(f"фото: {len(all_files)}, пар: {len(pairs)}, модель: {service.model_version}")
    cache = root.parent / f"{args.name}_{service.model_version}.npz"
    emb, quality, reasons = embed_all(service, root, all_files, cache)
    print(f"вектор есть: {len(emb)}, без пригодного лица: {len(reasons)} {dict(Counter(reasons.values()))}")

    # ---- верификация по парам ----
    ok = [(a, b, s, f) for a, b, s, f in pairs if a in emb and b in emb]
    sims = np.array([float(emb[a] @ emb[b]) for a, b, _, _ in ok])
    same = np.array([s == 1 for _, _, s, _ in ok])
    folds = np.array([f for *_, f in ok])
    grid = np.round(np.arange(-0.2, 1.0, 0.005), 3)
    fold_acc, fold_thr = [], []
    for k in range(n_folds):                     # порог подбираем на 9 фолдах, проверяем на 10-м
        tr, te = folds != k, folds == k
        t = best_threshold(sims[tr], same[tr], grid)
        fold_thr.append(t)
        fold_acc.append(float(np.mean((sims[te] >= t) == same[te])))
    curve = [(float(t), *far_frr(sims, same, t)) for t in grid]
    eer_t, eer_far, eer_frr = min(curve, key=lambda c: abs(c[1] - c[2]))

    def thr_at_far(target):
        t = next((t for t, far, _ in curve if far <= target), 1.0)
        return t, *far_frr(sims, same, t)

    at = {f"FAR<={x:g}": dict(zip(("threshold", "far", "frr"), map(lambda v: round(v, 4), thr_at_far(x))))
          for x in (0.01, 0.001)}

    # ---- опознание как у BE: база лиц (первые N фото), запросы — остальные ----
    by_person = defaultdict(list)
    for f in all_files:
        if f in emb:
            by_person[f.split("/")[0]].append(f)
    people = {pid: fs for pid, fs in by_person.items() if len(fs) >= args.gallery + 1}
    g_ids, g_vecs = [], []
    for pid, fs in people.items():
        for f in fs[:args.gallery]:
            g_ids.append(pid)
            g_vecs.append(emb[f])
    g_vecs = np.stack(g_vecs)
    g_ids = np.array(g_ids)
    probe_thr = at["FAR<=0.001"]["threshold"]
    ident = Counter()
    for pid, fs in people.items():
        for f in fs[args.gallery:]:
            s = g_vecs @ emb[f]
            j = int(np.argmax(s))
            ident["rank1_correct" if g_ids[j] == pid else "rank1_wrong"] += 1
            if s[j] < probe_thr:
                ident["no_match_at_thr"] += 1
            elif g_ids[j] == pid:
                ident["correct_at_thr"] += 1
            else:
                ident["wrong_at_thr"] += 1
    n_probe = ident["rank1_correct"] + ident["rank1_wrong"]

    report = {
        "dataset": args.name, "embedding_model_version": service.model_version,
        "images": len(all_files), "embedded": len(emb), "no_usable_face": dict(Counter(reasons.values())),
        "pairs_used": int(len(ok)), "pairs_skipped": len(pairs) - len(ok),
        "verification": {
            "accuracy_10fold_mean": round(float(np.mean(fold_acc)), 4),
            "accuracy_10fold_std": round(float(np.std(fold_acc)), 4),
            "threshold_10fold_mean": round(float(np.mean(fold_thr)), 3),
            "eer": round((eer_far + eer_frr) / 2, 4), "eer_threshold": eer_t,
            "same_sim_mean": round(float(sims[same].mean()), 3),
            "diff_sim_mean": round(float(sims[~same].mean()), 3),
            **at,
        },
        "identification": {
            "people_in_gallery": len(people), "gallery_photos": int(len(g_ids)), "probes": n_probe,
            "rank1_accuracy": round(ident["rank1_correct"] / max(n_probe, 1), 4),
            "threshold": probe_thr,
            "correct_at_thr": ident["correct_at_thr"], "wrong_at_thr": ident["wrong_at_thr"],
            "no_match_at_thr": ident["no_match_at_thr"],
        },
        "note": "LFW — проверка работы сервиса; рабочий порог — на своих добровольцах и камерах (M6)",
    }
    (out_dir / f"{args.name}_calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.hist(sims[same], bins=60, alpha=0.6, label="тот же человек")
    a1.hist(sims[~same], bins=60, alpha=0.6, label="разные люди")
    for name, v in at.items():
        a1.axvline(v["threshold"], ls="--", c="k" if "0.001" in name else "gray", label=f"порог {name}")
    a1.set_xlabel("косинусное сходство"); a1.set_ylabel("пар"); a1.legend(); a1.set_title(f"{args.name}: сходство пар")
    ts = [c[0] for c in curve]
    a2.plot(ts, [c[1] for c in curve], label="FAR (чужой принят)")
    a2.plot(ts, [c[2] for c in curve], label="FRR (свой не узнан)")
    a2.set_yscale("log"); a2.set_ylim(1e-4, 1); a2.set_xlabel("порог"); a2.legend(); a2.set_title("FAR / FRR")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.name}_calibration.png", dpi=110)
    print(f"-> {out_dir / (args.name + '_calibration.json')}, .png")


if __name__ == "__main__":
    main()
