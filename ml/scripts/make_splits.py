"""D1: реестр данных + разметка событий + dev/test без утечек + отчёт о данных.

    python scripts/make_splits.py                 # всё из data/raw -> data/manifest.csv, data/annotations, data/splits
    python scripts/make_splits.py --sha           # + sha256 каждого файла (долго на RWF-2000, ~12 ГБ)

Пишет (это всё в git, сами видео — нет):
  data/manifest.csv                  — ролик, источник, лицензия, задача, метка, группа, split, теги
  data/annotations/events.csv        — интервалы инцидентов из разметки датасетов
                                       (+ data/annotations/manual.csv — ручная разметка, не перезаписывается)
  data/splits/<task>_{dev,test}_v1.txt — test заморожен: если файл есть, он не меняется
  reports/data_report.md             — состав данных по задачам и split, лицензии
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.eval.datasets import Interval, VideoRow, collect, read_intervals, sha256_of, write_csv  # noqa: E402
from vision.eval.splits import assign, check_no_leak, read_frozen, write_split_lists  # noqa: E402

DATA = ROOT / "data"


def data_report(rows: list[VideoRow], ivs: list[Interval]) -> str:
    lines = ["# Данные для оценки (D1)", "",
             "Сгенерировано `scripts/make_splits.py`. dev — настраиваем пороги; test — заморожен, "
             "смотрим один раз в конце (ТЗ 5.2). Делим по группам (человек/сцена), а не по кадрам.", "",
             "| задача | split | роликов | с инцидентом | без | интервалов | часов видео | источники |",
             "|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r.task, r.split)].append(r)
    iv_count = Counter(i.video_id for i in ivs)
    for (task, split), rs in sorted(by.items()):
        pos = sum(r.label == "positive" for r in rs)
        n_iv = sum(iv_count[r.video_id] for r in rs)
        hours = sum(r.duration_s for r in rs) / 3600
        src = ", ".join(sorted({r.source for r in rs}))
        lines.append(f"| {task} | {split} | {len(rs)} | {pos} | {len(rs) - pos} | {n_iv} | {hours:.2f} | {src} |")
    lines += ["", "## Теги (сцены)", "", "| задача | тег | роликов |", "|---|---|---|"]
    tags = Counter((r.task, t) for r in rows for t in r.tags.split(";") if t)
    for (task, t), n in sorted(tags.items()):
        lines.append(f"| {task} | {t} | {n} |")
    lines += ["", "## Лицензии", ""]
    for src, lic in sorted({(r.source, r.license) for r in rows}):
        lines.append(f"- **{src}** — {lic}")
    lines += ["", "## Ограничения", "",
              "- Всё это публичные клипы с чужих камер. Метрики на них — не метрики на камерах стенда: "
              "финальный test должен включать постановочные съёмки (D4, `docs/shooting_plan.md`).",
              "- UR Fall: клипы обрываются через 0.1–2 с после падения, а правило ждёт `still_s` неподвижности — "
              "при оценке последний кадр удерживается (`--hold-s`, «человек продолжает лежать»).",
              "- RWF-2000 и D-Fire Videos размечены на уровне клипа (весь клип — инцидент или нет).",
              "- Зона: готовой разметки нет — ручная разметка в `data/annotations/manual.csv`."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="", help="через запятую: urfall,rwf2000,dfire (по умолчанию все)")
    p.add_argument("--test-ratio", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sha", action="store_true", help="посчитать sha256 файлов")
    args = p.parse_args()

    rows, ivs = collect(ROOT, [s for s in args.sources.split(",") if s] or None)
    if not rows:
        p.error("в data/raw нет известных датасетов")
    frozen = read_frozen(DATA / "splits")
    assign(rows, args.test_ratio, args.seed, frozen)
    check_no_leak(rows)
    if args.sha:
        for i, r in enumerate(rows, 1):
            r.sha256 = sha256_of(ROOT / r.file)
            if i % 200 == 0:
                print(f"  sha256: {i}/{len(rows)}", flush=True)

    write_csv(DATA / "manifest.csv", rows, VideoRow.columns())
    write_csv(DATA / "annotations" / "events.csv", ivs, Interval.columns())
    manual = DATA / "annotations" / "manual.csv"
    if not manual.exists():
        write_csv(manual, [], Interval.columns())
    written = write_split_lists(rows, DATA / "splits")
    (ROOT / "reports").mkdir(exist_ok=True)
    report = data_report(rows, ivs + read_intervals(manual))
    (ROOT / "reports" / "data_report.md").write_text(report, encoding="utf-8")

    print(f"роликов: {len(rows)}, интервалов: {len(ivs)}, заморожено test: {sorted(frozen) or 'нет (создан v1)'}")
    print("списки:", ", ".join(p.name for p in written))
    print(report)


if __name__ == "__main__":
    main()
