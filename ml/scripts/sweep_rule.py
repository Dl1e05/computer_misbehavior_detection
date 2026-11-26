"""Подбор порогов правила на dev (шаг «коррекция перед обучением»): несколько настроек -> одна таблица.

    python scripts/sweep_rule.py --task fall --config "base:fall.bbox_fallback=false" \\
        --config "fallback:" --config "win1.5:fall.fall_window_s=1.5"
    python scripts/sweep_rule.py --task fight --limit 100 --config "a:fight.suspect_thr=0.4" ...

Каждая настройка — `имя:правило.параметр=значение;правило.параметр=значение` (пусто — как в cameras.yaml).
Гоняет scripts/eval_events.py ТОЛЬКО на dev (test не трогаем — ТЗ 5.2), складывает отчёты в reports/eval/
и пишет сводку reports/eval/<task>_tuning_<время>.md: precision, recall, F1, ложные в час, задержка,
отдельно по тегам (например, сбоку / сверху).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(task: str, name: str, sets: list[str], args) -> dict:
    tag = f"sweep-{name}"
    cmd = [sys.executable, str(ROOT / "scripts/eval_events.py"), "--task", task, "--split", "dev", "--tag", tag]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    for s in sets:
        cmd += ["--set", s]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("-> ")), None)
    if line is None:
        raise SystemExit(f"{name}: прогон не удался:\n{(p.stderr or p.stdout)[-1500:]}")
    report = Path(line[3:].split("  (")[0].strip()).with_suffix(".json")
    return json.loads(report.read_text(encoding="utf-8")) | {"_file": report.stem}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--config", action="append", required=True, help="имя:правило.параметр=значение;...")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows = []
    for spec in args.config:
        name, _, rest = spec.partition(":")
        sets = [x.strip() for x in rest.split(";") if x.strip()]
        print(f"[{name}] {sets or 'как в cameras.yaml'} ...", flush=True)
        rep = run(args.task, name, sets, args)
        for level, res in rep["levels"].items():
            o = res["overall"]
            tags = {t: s.get("recall") for t, s in res["by_tag"].items() if s.get("intervals")}
            rows.append((name, ";".join(sets) or "—", level, o, tags, rep["_file"]))
            print(f"    {level}: P {o['precision']} R {o['recall']} F1 {o['f1']} FP {o['fp']} "
                  f"({o['fp_per_hour']}/ч) задержка {o['delay_p50_s']} | recall по тегам {tags}", flush=True)

    out = ROOT / "reports/eval" / f"{args.task}_tuning_{datetime.now():%Y%m%d-%H%M}.md"
    lines = [f"# Подбор порогов `{args.task}` на dev — {datetime.now():%Y-%m-%d %H:%M}", "",
             "Только dev (test заморожен). Отчёты прогонов — по ссылкам.", "",
             "| настройка | параметры | уровень | precision | recall | F1 | FP | ложных в час | задержка p50 | recall по тегам | отчёт |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sets, level, o, tags, f in rows:
        tg = ", ".join(f"{k} {v}" for k, v in tags.items())
        lines.append(f"| {name} | `{sets}` | {level} | {o['precision']} | {o['recall']} | {o['f1']} | {o['fp']} | "
                     f"{o['fp_per_hour']} | {o['delay_p50_s']} | {tg} | [{f}]({f}.md) |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
