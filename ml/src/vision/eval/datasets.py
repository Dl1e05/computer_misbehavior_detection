"""Реестр данных для оценки по событиям (шаг 10, D1).

Собирает из data/raw/ строки манифеста (один ролик — одна строка) и разметку событий
(интервалы «инцидент был с start_s по end_s»). Сейчас поддержаны:
- UR Fall: падения (cam0 сбоку, cam1 сверху, покадровая разметка авторов) и ADL — негативы;
- RWF-2000: драка / не драка на уровне клипа (у авторов свой train/val -> наш dev/test);
- D-Fire Videos: VP — огонь/дым на уровне клипа, FP — похожее без огня.
Новые источники (постановочные съёмки, ручная разметка) добавляются функцией того же вида.

Пути — относительно папки ml/ (так манифест одинаков на обеих машинах).
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2

URFALL_LICENSE = "CC BY-NC-SA 4.0 (только некоммерческое)"
RWF_LICENSE = "RWF-2000: только исследования, без распространения и изменений (условия авторов)"
DFIRE_LICENSE = "D-Fire: см. LICENSE репозитория gaiasd/DFireDataset"


@dataclass
class VideoRow:
    video_id: str
    file: str                     # относительно ml/
    source: str                   # urfall / rwf2000 / dfire / staged / ...
    license: str
    task: str                     # fall / fight / fire / zone_intrusion / ...
    label: str                    # positive (есть инцидент) / negative
    group_id: str                 # всё с одним group_id — в одном split (один человек/дубль/сцена)
    duration_s: float
    fps: float
    split: str = ""               # dev / test (заполняет splits.assign)
    predefined_split: str = ""    # если split задан авторами датасета
    tags: str = ""                # через ';': side, overhead, night, group, occlusion, adl_lying ...
    crop: str = ""                # "x1,y1,x2,y2" в долях, если нужна часть кадра
    sha256: str = ""

    @classmethod
    def columns(cls) -> list[str]:
        return [f.name for f in fields(cls)]


@dataclass
class Interval:
    video_id: str
    type: str                     # incident_type
    start_s: float
    end_s: float
    to_end: bool = False          # интервал длится до конца ролика (на hold_last_s продлевается)
    origin: str = "dataset"       # dataset / manual / journal

    @classmethod
    def columns(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def video_meta(path: Path) -> tuple[float, float]:
    """(длительность, fps) по метаданным файла."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return round(n / fps, 3), round(fps, 3)


def short_id(prefix: str, rel: str) -> str:
    return f"{prefix}-{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:10]}"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


# ---------------- UR Fall ----------------

def _urfall_labels(csv_path: Path) -> dict[str, list[tuple[int, int]]]:
    seq: dict[str, list[tuple[int, int]]] = defaultdict(list)
    if csv_path.exists():
        for r in csv.reader(csv_path.open()):
            if len(r) >= 3:
                seq[r[0]].append((int(r[1]), int(r[2])))
    return {k: sorted(v) for k, v in seq.items()}


def urfall(root: Path, ml: Path) -> tuple[list[VideoRow], list[Interval]]:
    """Падение = от первого кадра «падает» (0) до конца клипа (дальше человек лежит)."""
    d = root / "urfall"
    if not d.exists():
        return [], []
    falls = _urfall_labels(d / "urfall-cam0-falls.csv")
    adls = _urfall_labels(d / "urfall-cam0-adls.csv")
    rows, ivs = [], []
    for f in sorted(d.glob("*.mp4")):
        name = f.stem                              # fall-01-cam0 / adl-05-cam0
        seq, cam = name.rsplit("-", 1)
        rel = f.relative_to(ml).as_posix()
        dur, fps = video_meta(f)
        view = "side" if cam == "cam0" else "overhead"
        if seq.startswith("fall"):
            labels = falls.get(seq, [])
            falling = [fr for fr, lab in labels if lab == 0]
            tags = [view]
            rows.append(VideoRow(name, rel, "urfall", URFALL_LICENSE, "fall", "positive", seq, dur, fps,
                                 tags=";".join(tags), crop="0.5,0,1,1"))
            if falling:
                ivs.append(Interval(name, "fall", round((falling[0] - 1) / fps, 3), dur, to_end=True))
        else:
            labels = adls.get(seq, [])
            tags = [view, "adl"] + (["adl_lying"] if any(lab == 1 for _, lab in labels) else [])
            rows.append(VideoRow(name, rel, "urfall", URFALL_LICENSE, "fall", "negative", seq, dur, fps,
                                 tags=";".join(tags), crop="0.5,0,1,1"))
    return rows, ivs


# ---------------- RWF-2000 ----------------

def rwf2000(root: Path, ml: Path) -> tuple[list[VideoRow], list[Interval]]:
    d = root / "rwf2000" / "RWF-2000" / "RWF-2000"
    if not d.exists():
        return [], []
    rows, ivs = [], []
    for split_dir, ours in (("train", "dev"), ("val", "test")):
        for cls_dir in sorted((d / split_dir).iterdir()):
            positive = not cls_dir.name.endswith("NonFight")
            for f in sorted(cls_dir.glob("*.avi")):
                rel = f.relative_to(ml).as_posix()
                vid = short_id("rwf", rel)
                dur, fps = video_meta(f)
                rows.append(VideoRow(vid, rel, "rwf2000", RWF_LICENSE, "fight",
                                     "positive" if positive else "negative", vid, dur, fps,
                                     predefined_split=ours, tags="cctv"))
                if positive:                       # у RWF драка на уровне клипа: весь клип
                    ivs.append(Interval(vid, "fight", 0.0, dur, to_end=True))
    return rows, ivs


# ---------------- D-Fire surveillance videos ----------------

def dfire_videos(root: Path, ml: Path) -> tuple[list[VideoRow], list[Interval]]:
    d = root / "d-fire" / "Videos"
    if not d.exists():
        return [], []
    rows, ivs = [], []
    for f in sorted(d.glob("*.mp4")):
        positive = f.stem.upper().startswith("VP")
        rel = f.relative_to(ml).as_posix()
        vid = f"dfire-{f.stem.lower()}"
        dur, fps = video_meta(f)
        rows.append(VideoRow(vid, rel, "dfire", DFIRE_LICENSE, "fire", "positive" if positive else "negative",
                             vid, dur, fps, tags="cctv" if positive else "cctv;hard_negative"))
        if positive:
            ivs.append(Interval(vid, "fire", 0.0, dur, to_end=True))
    return rows, ivs


SOURCES = {"urfall": urfall, "rwf2000": rwf2000, "dfire": dfire_videos}


def collect(ml: Path, sources: list[str] | None = None) -> tuple[list[VideoRow], list[Interval]]:
    root = ml / "data" / "raw"
    rows, ivs = [], []
    for name, fn in SOURCES.items():
        if sources and name not in sources:
            continue
        r, i = fn(root, ml)
        rows += r
        ivs += i
    return rows, ivs


# ---------------- чтение / запись ----------------

def write_csv(path: Path, items, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for it in items:
            w.writerow(asdict(it))


def read_manifest(path: Path) -> list[VideoRow]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["duration_s"], r["fps"] = float(r["duration_s"]), float(r["fps"])
            out.append(VideoRow(**r))
    return out


def read_intervals(path: Path) -> list[Interval]:
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(Interval(r["video_id"], r["type"], float(r["start_s"]), float(r["end_s"]),
                                r.get("to_end", "False") == "True", r.get("origin", "dataset")))
    return out
