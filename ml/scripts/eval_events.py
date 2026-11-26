"""Оценка по событиям (шаг 10, Фаза 4.1): конвейер на размеченных роликах -> precision / recall / F1,
ложные в час, задержка, confusion matrix по клипам, разбивка по сценам (тегам), таблица ошибок.

    python scripts/eval_events.py --task fall                        # dev UR Fall
    python scripts/eval_events.py --task fight --limit 100           # 100 драк + 100 не-драк из dev RWF-2000
    python scripts/eval_events.py --task fight --split test          # test — ОДИН раз в конце (пишется в журнал)
    python scripts/eval_events.py --task fall --set fall.still_s=2   # другой порог правила (подбор на dev)
    python scripts/eval_events.py --task fight --label negative --rule running,lying_motionless,camera_tampering
        # ложные в час новых правил на обычном видео (ТЗ 18): ролики задачи fight без драк

Тот же скрипт сравнивает базу (эвристики) и обученные модели — «было / стало» честно (Фаза 4).
Ролики идут пачками как камеры одного узла (тот же Pipeline, что в демо), файлы — без ожидания и
без пропуска кадров. Лица и MQTT выключены.
Отчёт: reports/eval/<task>_<split>_<время>.{json,md} + события в .events.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.eval.datasets import read_intervals, read_manifest  # noqa: E402
from vision.eval.matching import match_video, summarize  # noqa: E402
from vision.runtime.bus import FrameBus  # noqa: E402
from vision.runtime.pipeline import Pipeline  # noqa: E402

LEVELS = {"fall": ["detection"], "fight": ["suspicion", "confirmation"], "fire": ["detection"]}
HOLD_S = {"fall": 5.0}   # удержание последнего кадра нужно только коротким клипам падений UR Fall
RULES_FROM = {"fall": "cam-fall", "fight": "cam-fight", "fire": "cam-fire"}   # параметры правила из cameras.yaml
DEFAULT_RULES_CAMERA = "cam-01"                                                # для остальных правил


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8")) or {}


def rule_params(task: str, camera: str | None, overrides: list[str]) -> dict:
    cams = {c["id"]: c for c in load_yaml("cameras.yaml").get("cameras", [])}
    cam = cams.get(camera or RULES_FROM.get(task, DEFAULT_RULES_CAMERA), {})
    params = dict((cam.get("rules") or {}).get(task) or {})
    for ov in overrides:                                   # fall.still_s=2 -> params["still_s"] = 2
        key, val = ov.split("=", 1)
        rule, _, name = key.partition(".")
        if rule == task:
            params[name] = yaml.safe_load(val)
    return params


def run_videos(rows, rules: dict, args, app) -> tuple[dict, dict, dict]:
    """rules {имя: параметры} -> события {video_id: [(incident, event_type, t, conf)]}, длительность, статус."""
    events: dict[str, list] = defaultdict(list)
    first_ts, last_t = {}, {}
    shared, status = None, {}
    for i in range(0, len(rows), args.chunk):
        chunk = rows[i:i + args.chunk]
        cams = [{"id": r.video_id, "source": r.file, "process_fps": args.fps, "realtime": False,
                 "crop": [float(x) for x in r.crop.split(",")] if r.crop else None,
                 "hold_last_s": args.hold_s, "rules": rules} for r in chunk]
        node = Pipeline(app, cams, {}, ROOT, fast=True, faces=False, mqtt=False, shared=shared)
        shared = shared or node
        status = node.status.snapshot()
        if not any(node.rules.values()):                     # правило выключено (нет весов огня)
            return {}, {}, status
        bus = FrameBus(node.sources, drop_old=False).start()   # без пропусков кадров
        try:
            while not bus.done:
                for o in node.process(bus.next_batch(timeout=1.0)):
                    vid = o.frame.camera_id
                    t0 = first_ts.setdefault(vid, o.frame.ts)
                    last_t[vid] = (o.frame.ts - t0).total_seconds()
                    for e in o.events:
                        events[vid].append((e.incident_type, e.event_type,
                                            round((e.occurred_at - t0).total_seconds(), 3), e.confidence))
        finally:
            bus.stop()
            node.close()
        print(f"  {min(i + args.chunk, len(rows))}/{len(rows)} роликов, событий: "
              f"{sum(len(v) for v in events.values())}", flush=True)
    return events, last_t, status


def evaluate(rows, ivs, events, task, args, multi: bool = False) -> dict:
    by_vid = defaultdict(list)
    for iv in ivs:
        if iv.type == task:
            by_vid[iv.video_id].append(iv)
    out = {}
    for level in LEVELS.get(task, ["detection"]):
        results = []
        for r in rows:
            dur = r.duration_s + args.hold_s
            intervals = [(iv.start_s, dur if iv.to_end else iv.end_s) for iv in by_vid.get(r.video_id, [])]
            ts = [t for inc, et, t, _ in events.get(r.video_id, []) if inc == task and et == level]
            results.append(match_video(r.video_id, dur, intervals, ts, args.before_s, args.after_s))
        tags = sorted({t for r in rows for t in r.tags.split(";") if t})
        by_tag = {}
        for tag in tags:
            ids = {r.video_id for r in rows if tag in r.tags.split(";")}
            by_tag[tag] = summarize([x for x in results if x.video_id in ids])
        rows_by_id = {r.video_id: r for r in rows}
        errors = {
            "missed": [{"video_id": x.video_id, "file": rows_by_id[x.video_id].file, "tags": rows_by_id[x.video_id].tags}
                       for x in results if x.fn],
            "false": [{"video_id": x.video_id, "file": rows_by_id[x.video_id].file, "tags": rows_by_id[x.video_id].tags,
                       "times_s": x.fp_times} for x in results if x.fp],
        }
        out[f"{task} / {level}" if multi else level] = {"overall": summarize(results), "by_tag": by_tag,
                                                         "errors": errors}
    return out


def to_md(report: dict) -> str:
    c = report["config"]
    lines = [f"# Оценка: {c['task']} / {c['split']} — {report['created']}", "",
             f"Роликов: {report['videos']} ({c['limit_note']}); process_fps {c['fps']}; удержание последнего кадра "
             f"{c['hold_s']} с; окно совпадения [начало − {c['before_s']} с, конец + {c['after_s']} с].",
             f"Правила: `{json.dumps(c['rule_params'], ensure_ascii=False)}`",
             f"Модели: `{c['model_version']}`", ""]
    for level, res in report["levels"].items():
        o = res["overall"]
        cm = o["clips"]
        title = level if " / " in level else f"{c['task']} / {level}"
        lines += [f"## {title}", "",
                  "| precision | recall | F1 | TP | FN | FP | дубли | ложных в час | задержка p50 / p95, с |",
                  "|---|---|---|---|---|---|---|---|---|",
                  f"| {o['precision']} | {o['recall']} | {o['f1']} | {o['tp']} | {o['fn']} | {o['fp']} | "
                  f"{o['duplicates']} | {o['fp_per_hour']} (на {o['calm_hours']} ч) | {o['delay_p50_s']} / {o['delay_p95_s']} |",
                  "", "Клипы (confusion matrix):", "",
                  "| | событие было | события не было |", "|---|---|---|",
                  f"| инцидент есть | {cm['pos_pred']} | {cm['pos_miss']} |",
                  f"| инцидента нет | {cm['neg_pred']} | {cm['neg_clean']} |", ""]
        if res["by_tag"]:
            lines += ["По сценам:", "", "| тег | роликов | precision | recall | FP | ложных в час |",
                      "|---|---|---|---|---|---|"]
            for tag, s in res["by_tag"].items():
                lines.append(f"| {tag} | {s['videos']} | {s['precision']} | {s['recall']} | {s['fp']} | {s['fp_per_hour']} |")
            lines.append("")
        miss, false = res["errors"]["missed"], res["errors"]["false"]
        lines += [f"Ошибки: пропусков {len(miss)}, роликов с ложными {len(false)} (полный список — в .json).", ""]
        if miss:
            lines += ["| пропуск | теги | файл |", "|---|---|---|"]
            lines += [f"| {m['video_id']} | {m['tags']} | `{m['file']}` |" for m in miss[:15]]
            lines.append("")
        if false:
            lines += ["| ложное | теги | моменты, с | файл |", "|---|---|---|---|"]
            lines += [f"| {f['video_id']} | {f['tags']} | {f['times_s']} | `{f['file']}` |" for f in false[:15]]
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(LEVELS))
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--limit", type=int, default=0, help="не больше N роликов каждого класса (0 — все)")
    p.add_argument("--fps", type=float, default=8.0, help="process_fps")
    p.add_argument("--hold-s", type=float, default=None,
                   help="удерживать последний кадр, с (по умолчанию 5 для fall, 0 для остальных)")
    p.add_argument("--before-s", type=float, default=1.0, help="событие может быть раньше начала на столько")
    p.add_argument("--after-s", type=float, default=2.0, help="и позже конца на столько")
    p.add_argument("--chunk", type=int, default=16, help="роликов одновременно (как камер одного узла)")
    p.add_argument("--rules-from", help="камера из cameras.yaml, чьи параметры правила взять")
    p.add_argument("--set", action="append", default=[], help="переопределить параметр: fall.still_s=2")
    p.add_argument("--tag", default="", help="метка в имени отчёта")
    p.add_argument("--rule", default="", help="какие правила гонять (через запятую); по умолчанию = --task")
    p.add_argument("--label", default="all", choices=["all", "positive", "negative"], help="какие ролики брать")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.hold_s is None:
        args.hold_s = HOLD_S.get(args.task, 0.0)

    manifest = ROOT / "data" / "manifest.csv"
    if not manifest.exists():
        p.error("нет data/manifest.csv — сначала python scripts/make_splits.py")
    rows = [r for r in read_manifest(manifest) if r.task == args.task and r.split == args.split
            and args.label in ("all", r.label)]
    if args.limit:
        pos = [r for r in rows if r.label == "positive"][:args.limit]
        neg = [r for r in rows if r.label == "negative"][:args.limit]
        rows = pos + neg
    if not rows:
        p.error(f"нет роликов {args.task}/{args.split}")
    ivs = read_intervals(ROOT / "data/annotations/events.csv") + read_intervals(ROOT / "data/annotations/manual.csv")
    if args.split == "test":
        print("ВНИМАНИЕ: test — смотрим один раз, после выбора порогов на dev. Запуск записан в "
              "reports/eval/test_runs.csv", flush=True)

    app = load_yaml("app.yaml")
    names = [x for x in args.rule.split(",") if x] or [args.task]
    rules = {n: rule_params(n, args.rules_from if n == args.task else None, args.set) for n in names}
    print(f"{args.task}/{args.split}: роликов {len(rows)}, правила {rules}", flush=True)
    t0 = time.monotonic()
    events, processed, status = run_videos(rows, rules, args, app)
    if not events and not processed:
        states = {c["camera_id"]: c["rules"] for c in status.get("cameras", [])}
        print(f"оценка невозможна: правила выключены: {next(iter(states.values()), {})}")
        return

    from vision.events.builder import ModelVersions
    from vision.rules import RULES

    report = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "videos": len(rows), "wall_s": round(time.monotonic() - t0, 1),
        "config": {"task": args.task, "split": args.split, "fps": args.fps, "hold_s": args.hold_s,
                   "before_s": args.before_s, "after_s": args.after_s, "rule_params": rules, "label": args.label,
                   "limit_note": f"по {args.limit} каждого класса" if args.limit else "все",
                   "model_version": "; ".join(ModelVersions(app.get("versions", {})).for_incident(
                       n, set(RULES[n].requires)) for n in names)},
        "levels": {k: v for n in names
                   for k, v in evaluate(rows, ivs, events, n, args, len(names) > 1 or n != args.task).items()},
    }
    out_dir = ROOT / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_{args.split}_{datetime.now():%Y%m%d-%H%M}{'_' + args.tag if args.tag else ''}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_md(report)
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    with (out_dir / f"{stem}.events.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video_id", "event_type", "t_s", "confidence"])
        for vid, evs in sorted(events.items()):
            for inc, et, t, conf in evs:
                w.writerow([vid, f"{inc}/{et}", t, conf])
    if args.split == "test":
        log = out_dir / "test_runs.csv"
        new = not log.exists()
        with log.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["when", "task", "report", "rule_params", "model_version"])
            w.writerow([report["created"], args.task, stem, json.dumps(rules, ensure_ascii=False),
                        report["config"]["model_version"]])
    print(md)
    print(f"-> {out_dir / stem}.md  ({report['wall_s']} с)")


if __name__ == "__main__":
    main()
