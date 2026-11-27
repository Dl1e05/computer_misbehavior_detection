"""M1: списки D-Fire для обучения огня/дыма — train / val / test, без копирования картинок.

    python scripts/prepare_fire_data.py            # -> data/splits/fire_img_*_v1.txt + data/processed/fire/v1/*.txt

У D-Fire есть только train и test. val режем из train сами, но НЕ случайно по картинкам:
AoF* — кадры из видео подряд (AoF06723, AoF06724, ...), соседние кадры почти одинаковые, и
случайный val был бы завышен (утечка). Поэтому делим по блокам из BLOCK соседних номеров
одного префикса (AoF / WEB / PublicDataset) — блок целиком уходит в train или в val.

test — авторский test D-Fire, не трогаем (ТЗ 5.2: смотрим один раз в конце).

Пишет:
  data/splits/fire_img_{train,val,test}_v1.txt — имена файлов (в git; val заморожен, если файл есть)
  data/processed/fire/v1/{train,val,test}.txt  — абсолютные пути картинок для Ultralytics
                                                 (метки он ищет сам: /images/ -> /labels/)
"""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "d-fire" / "D-Fire"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "data" / "processed" / "fire" / "v1"
NAME = re.compile(r"^([A-Za-z]+?)(\d+)\.jpg$")


def group_of(name: str, block: int) -> str:
    m = NAME.match(name)
    return f"{m.group(1)}:{int(m.group(2)) // block}" if m else name


def classes_of(label: Path) -> set[int]:
    return {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}


def stats(names: list[str], part: str) -> str:
    c = Counter()
    for n in names:
        cls = classes_of(RAW / part / "labels" / (n[:-4] + ".txt"))
        c["smoke"] += 0 in cls
        c["fire"] += 1 in cls
        c["empty"] += not cls
    return f"{len(names)} картинок: с дымом {c['smoke']}, с огнём {c['fire']}, фон {c['empty']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.15, help="доля train-блоков в val")
    ap.add_argument("--block", type=int, default=100, help="соседних номеров в одном блоке")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_all = sorted(p.name for p in (RAW / "train" / "images").glob("*.jpg"))
    test = sorted(p.name for p in (RAW / "test" / "images").glob("*.jpg"))

    frozen = SPLITS / "fire_img_val_v1.txt"
    if frozen.exists():                                  # val однажды выбран — не меняется
        val = set(frozen.read_text().split())
    else:
        # стратификация: фон в D-Fire лежит блоками номеров, поэтому блоки делим на «в основном фон» /
        # «в основном огонь-дым» отдельно по каждому префиксу — иначе в val мало фона и ложные не видны
        members: dict[str, list[str]] = {}
        for n in train_all:
            members.setdefault(group_of(n, args.block), []).append(n)
        strata: dict[tuple, list[str]] = {}
        for g, ns in sorted(members.items()):
            empty = sum(not classes_of(RAW / "train" / "labels" / (n[:-4] + ".txt")) for n in ns)
            strata.setdefault((g.split(":")[0], empty * 2 > len(ns)), []).append(g)
        rng = random.Random(args.seed)
        val_groups = set()
        for key in sorted(strata):
            gs = strata[key]
            rng.shuffle(gs)
            val_groups |= set(gs[: round(len(gs) * args.val)])
        val = {n for n in train_all if group_of(n, args.block) in val_groups}
    train = [n for n in train_all if n not in val]
    val_l = sorted(val)

    # проверка на утечку: ни один блок не попал в обе части
    leak = {group_of(n, args.block) for n in train} & {group_of(n, args.block) for n in val_l}
    assert not leak, f"блоки в train и val одновременно: {sorted(leak)[:5]}"

    SPLITS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, names, part in (("train", train, "train"), ("val", val_l, "train"), ("test", test, "test")):
        lst = SPLITS / f"fire_img_{split}_v1.txt"
        if not lst.exists():
            lst.write_text("\n".join(names) + "\n")
        (OUT / f"{split}.txt").write_text(
            "\n".join(str((RAW / part / "images" / n).resolve()) for n in names) + "\n")
        print(f"{split}: {stats(names, part)}")


if __name__ == "__main__":
    main()
