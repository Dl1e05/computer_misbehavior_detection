"""Разбиение на dev / test без утечек (ТЗ 5.2, D1, D6).

- Делим по group_id (человек / дубль / исходная сцена), а не по ролику и тем более не по кадрам:
  cam0 и cam1 одного падения UR Fall — одна группа, обе в одном split.
- Если split задан авторами (RWF-2000: train -> dev, val -> test) — берём его.
- Иначе — детерминированно по хэшу (seed, group_id): тот же результат на любой машине.
- Замороженный test (data/splits/<task>_test_v1.txt) не меняется: ролики из него всегда test,
  новые группы, которых там нет, идут в dev (новый test — это test_v2, старый остаётся).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from vision.eval.datasets import VideoRow


def _unit(seed: int, key: str) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def assign(rows: list[VideoRow], test_ratio: float = 0.4, seed: int = 0,
           frozen_test: dict[str, set[str]] | None = None) -> list[VideoRow]:
    """Проставляет row.split. frozen_test: {task: {video_id в замороженном test}}."""
    frozen_test = frozen_test or {}
    group_split: dict[tuple[str, str], str] = {}
    for r in rows:
        key = (r.task, r.group_id)
        frozen = frozen_test.get(r.task)
        if frozen is not None:
            split = "test" if r.video_id in frozen else "dev"
        elif r.predefined_split:
            split = r.predefined_split
        else:
            split = "test" if _unit(seed, f"{r.task}:{r.group_id}") < test_ratio else "dev"
        prev = group_split.setdefault(key, split)
        if prev != split:
            raise ValueError(f"группа {key} попала и в {prev}, и в {split} — утечка между dev и test")
        r.split = split
    return rows


def check_no_leak(rows: list[VideoRow]) -> None:
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        seen[(r.task, r.group_id)].add(r.split)
    bad = [k for k, v in seen.items() if len(v) > 1]
    if bad:
        raise ValueError(f"утечка: группы в обоих split: {bad[:5]}")


def read_frozen(splits_dir: Path, version: str = "v1") -> dict[str, set[str]]:
    out = {}
    for f in splits_dir.glob(f"*_test_{version}.txt"):
        task = f.name[: -len(f"_test_{version}.txt")]
        out[task] = {x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()}
    return out


def write_split_lists(rows: list[VideoRow], splits_dir: Path, version: str = "v1") -> list[Path]:
    """dev-список пишется всегда, test — только если ещё не заморожен."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    written = []
    by: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        by[(r.task, r.split)].append(r.video_id)
    for (task, split), ids in sorted(by.items()):
        path = splits_dir / f"{task}_{split}_{version}.txt"
        if split == "test" and path.exists():
            continue                                   # замороженный test не переписываем
        path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
        written.append(path)
    return written
