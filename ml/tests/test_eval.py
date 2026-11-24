"""Шаг 10: сопоставление событий с разметкой, метрики, dev/test без утечек (без GPU и без видео)."""
from __future__ import annotations

import pytest

from vision.eval.datasets import VideoRow
from vision.eval.matching import match_video, summarize
from vision.eval.splits import assign, check_no_leak, read_frozen, write_split_lists


def row(vid, group, task="fall", label="positive", predefined=""):
    return VideoRow(vid, f"data/{vid}.mp4", "src", "lic", task, label, group, 10.0, 30.0,
                    predefined_split=predefined)


# ---------- сопоставление ----------

def test_event_inside_interval_is_tp_with_delay():
    r = match_video("v", 10, [(2.0, 6.0)], [3.5])
    assert (r.tp, r.fn, r.fp) == (1, 0, 0) and r.delays == [1.5]


def test_tolerance_before_and_after():
    assert match_video("v", 20, [(5, 8)], [4.2], before_s=1).tp == 1      # чуть раньше начала
    assert match_video("v", 20, [(5, 8)], [9.5], after_s=2).tp == 1       # чуть позже конца
    assert match_video("v", 20, [(5, 8)], [11.0], after_s=2).fp == 1      # сильно позже — ложное


def test_duplicates_and_false_and_missed():
    r = match_video("v", 30, [(2, 5), (20, 25)], [3, 4, 12])
    assert (r.tp, r.fn, r.fp, r.dup) == (1, 1, 1, 1)
    assert r.fp_times == [12]


def test_calm_time_excludes_incidents():
    r = match_video("v", 60, [(10, 20), (15, 30)], [])
    assert r.calm_s == pytest.approx(40)                                  # 60 − объединение [10, 30]


def test_summary_metrics_and_confusion():
    res = [match_video("p1", 10, [(1, 9)], [2]),          # найдено
           match_video("p2", 10, [(1, 9)], []),           # пропуск
           match_video("n1", 3600, [], [100]),            # ложное на час спокойного видео
           match_video("n2", 3600, [], [])]
    s = summarize(res)
    assert (s["tp"], s["fn"], s["fp"]) == (1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5
    assert s["fp_per_hour"] == pytest.approx(1 / (7200 + 4) * 3600, rel=1e-3)
    assert s["clips"] == {"pos_pred": 1, "pos_miss": 1, "neg_pred": 1, "neg_clean": 1}


def test_summary_without_events_has_no_precision():
    s = summarize([match_video("p", 10, [(1, 2)], [])])
    assert s["precision"] is None and s["recall"] == 0.0


# ---------- dev / test ----------

def test_split_keeps_group_together_and_is_deterministic():
    rows = [row(f"fall-{i:02d}-cam{c}", f"fall-{i:02d}") for i in range(40) for c in (0, 1)]
    a = [r.split for r in assign(rows, 0.4, seed=0)]
    b = [r.split for r in assign([row(r.video_id, r.group_id) for r in rows], 0.4, seed=0)]
    assert a == b
    check_no_leak(rows)
    for i in range(40):
        assert rows[2 * i].split == rows[2 * i + 1].split                  # cam0 и cam1 — вместе
    assert 0.2 < sum(r.split == "test" for r in rows) / len(rows) < 0.6


def test_predefined_split_respected():
    rows = assign([row("a", "a", "fight", predefined="test"), row("b", "b", "fight", predefined="dev")], 0.5)
    assert [r.split for r in rows] == ["test", "dev"]


def test_leak_detected():
    rows = [row("a", "g", predefined="dev"), row("b", "g", predefined="test")]
    with pytest.raises(ValueError):
        assign(rows)


def test_frozen_test_is_not_rewritten(tmp_path):
    rows = assign([row(f"v{i}", f"g{i}") for i in range(20)], 0.5, seed=1)
    write_split_lists(rows, tmp_path)
    frozen = read_frozen(tmp_path)
    before = (tmp_path / "fall_test_v1.txt").read_text()
    rows2 = assign([row(f"v{i}", f"g{i}") for i in range(20)] + [row("new", "gnew")], 0.5, seed=99,
                   frozen_test=frozen)                                      # другой seed, новые данные
    write_split_lists(rows2, tmp_path)
    assert (tmp_path / "fall_test_v1.txt").read_text() == before
    assert {r.video_id for r in rows2 if r.split == "test"} == frozen["fall"]
    assert next(r for r in rows2 if r.video_id == "new").split == "dev"     # новое — не в старый test
