"""Сопоставление событий с разметкой и метрики по событиям (ТЗ 15.2, Фаза 4.1).

Для каждого ролика: интервалы инцидента [start, end] и события модели (момент t).
- событие попало в [start - before, end + after] -> интервал найден (TP); первое такое событие
  даёт задержку t - start; остальные события на тот же интервал — дубли (их склеивает BE);
- событие вне всех интервалов -> ложное (FP);
- интервал без событий -> пропуск (FN).
precision = TP / (TP + FP), recall = TP / (TP + FN); ложные в час — FP на час «спокойного» видео
(длительность роликов минус интервалы инцидентов).
Плюс по роликам (confusion matrix): ролик с инцидентом / без × было событие / не было.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VideoResult:
    video_id: str
    duration_s: float
    intervals: list[tuple[float, float]]
    events: list[float]
    tp: int = 0
    fn: int = 0
    fp: int = 0
    dup: int = 0
    delays: list[float] = field(default_factory=list)
    fp_times: list[float] = field(default_factory=list)
    calm_s: float = 0.0

    @property
    def positive(self) -> bool:
        return bool(self.intervals)

    @property
    def predicted(self) -> bool:
        return bool(self.events)


def _union_length(intervals: list[tuple[float, float]], lo: float, hi: float) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted((max(lo, s), min(hi, e)) for s, e in intervals):
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def match_video(video_id: str, duration_s: float, intervals: list[tuple[float, float]], events: list[float],
                before_s: float = 1.0, after_s: float = 2.0) -> VideoResult:
    res = VideoResult(video_id, duration_s, sorted(intervals), sorted(events))
    hit: dict[int, list[float]] = defaultdict(list)
    for t in res.events:
        idx = next((i for i, (s, e) in enumerate(res.intervals) if s - before_s <= t <= e + after_s), None)
        if idx is None:
            res.fp += 1
            res.fp_times.append(t)
        else:
            hit[idx].append(t)
    for i, (s, _e) in enumerate(res.intervals):
        if hit.get(i):
            res.tp += 1
            res.dup += len(hit[i]) - 1
            res.delays.append(max(0.0, hit[i][0] - s))
        else:
            res.fn += 1
    res.calm_s = max(0.0, duration_s - _union_length(res.intervals, 0.0, duration_s))
    return res


def summarize(results: list[VideoResult]) -> dict:
    tp = sum(r.tp for r in results)
    fn = sum(r.fn for r in results)
    fp = sum(r.fp for r in results)
    dup = sum(r.dup for r in results)
    calm_h = sum(r.calm_s for r in results) / 3600
    delays = [d for r in results for d in r.delays]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if tp + fp + fn else None)
    cm = {"pos_pred": 0, "pos_miss": 0, "neg_pred": 0, "neg_clean": 0}
    for r in results:
        if r.positive:
            cm["pos_pred" if r.predicted else "pos_miss"] += 1
        else:
            cm["neg_pred" if r.predicted else "neg_clean"] += 1
    rnd = lambda v: None if v is None else round(v, 3)  # noqa: E731
    return {
        "videos": len(results), "videos_positive": sum(r.positive for r in results),
        "intervals": tp + fn, "tp": tp, "fn": fn, "fp": fp, "duplicates": dup,
        "precision": rnd(precision), "recall": rnd(recall), "f1": rnd(f1),
        "calm_hours": round(calm_h, 3),
        "fp_per_hour": round(fp / calm_h, 2) if calm_h > 0 else None,
        "delay_p50_s": rnd(float(np.percentile(delays, 50))) if delays else None,
        "delay_p95_s": rnd(float(np.percentile(delays, 95))) if delays else None,
        "clips": cm,
    }
