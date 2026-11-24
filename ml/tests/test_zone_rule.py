"""Правило входа в зону на синтетических треках — без видео и GPU."""
from datetime import datetime, timedelta, timezone

import pytest

from vision.analytics.tracker import Track
from vision.analytics.zones import Zone, foot_point, load_zones, save_zones
from vision.rules import RULES, RuleContext, build_rules

W, H, FPS = 1000, 1000, 8
T0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
# зона — правая половина кадра
ZONE = Zone("zone-1", "cam-01", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)], version=1)


def track(tid, foot_x, conf=0.9):
    """Человек ростом 200 px, ноги в точке (foot_x, 500) пикселей."""
    return Track(tid, (foot_x - 20, 300, foot_x + 20, 500), conf, 0)


def rule(**params):
    return RULES["zone_intrusion"]("cam-01", {"zones": [ZONE], **params})


def run(r, xs_per_frame, start=0):
    """xs_per_frame: на каждом кадре словарь {track_id: x ног}. Возвращает события по кадрам."""
    events = []
    for i, xs in enumerate(xs_per_frame):
        ctx = RuleContext("cam-01", T0 + timedelta(seconds=(start + i) / FPS), W, H,
                          [track(tid, x) for tid, x in xs.items()])
        events.append(r.update(ctx))
    return events


def flat(events):
    return [e for frame in events for e in frame]


def test_foot_point_is_bottom_center():
    assert foot_point((100, 200, 300, 600), 1000, 1000) == (0.2, 0.6)


def test_walk_in_gives_one_event():
    r = rule(min_inside_s=0.5)
    # 4 кадра снаружи, потом 20 кадров внутри
    ev = run(r, [{1: 300}] * 4 + [{1: 700}] * 20)
    got = flat(ev)
    assert len(got) == 1
    e = got[0]
    assert e.incident_type == "zone_intrusion" and e.event_type == "detection"
    assert e.zone_id == "zone-1" and e.zone_version == 1
    assert e.participants[0].track_id == 1
    assert all(0 <= v <= 1 for v in e.participants[0].bbox)
    # событие не раньше, чем через 0.5 с внутри (4 кадра при 8 кадр/с)
    first = next(i for i, f in enumerate(ev) if f)
    assert first - 4 >= 4


def test_short_step_inside_is_not_entry():
    r = rule(min_inside_s=0.5)
    ev = run(r, [{1: 300}] * 4 + [{1: 700}] * 2 + [{1: 300}] * 10)   # 0.25 с внутри
    assert flat(ev) == []


def test_standing_on_border_jitter_gives_one_event():
    r = rule(min_inside_s=0.5, min_outside_s=1.0)
    ev = run(r, [{1: 300}] * 4 + [{1: 700}] * 8)                       # вошёл
    jitter = [{1: 495 if i % 2 else 505} for i in range(40)]          # дрожит на границе 5 с
    ev += run(r, jitter, start=12)
    assert len(flat(ev)) == 1


def test_exit_and_reenter_gives_second_event():
    r = rule(min_inside_s=0.5, min_outside_s=1.0)
    frames = [{1: 300}] * 4 + [{1: 700}] * 8 + [{1: 300}] * 12 + [{1: 700}] * 8
    assert len(flat(run(r, frames))) == 2


def test_spawn_inside_param():
    assert len(flat(run(rule(spawn_inside_counts=True), [{1: 700}] * 10))) == 1
    assert flat(run(rule(spawn_inside_counts=False), [{1: 700}] * 10)) == []


def test_two_tracks_entering_together_one_event_both_participants():
    r = rule(min_inside_s=0.5)
    ev = flat(run(r, [{1: 300, 2: 350}] * 4 + [{1: 700, 2: 750}] * 8))
    assert len(ev) == 1
    assert {p.track_id for p in ev[0].participants} == {1, 2}


def test_other_people_inside_are_participants():
    r = rule(min_inside_s=0.5)
    run(r, [{1: 700}] * 10)                                           # 1-й уже внутри (событие было)
    ev = flat(run(r, [{1: 700, 2: 300}] * 4 + [{1: 700, 2: 700}] * 8, start=10))
    assert len(ev) == 1
    assert [p.track_id for p in ev[0].participants] == [2, 1]         # вошедший первым


def test_threshold_in_seconds_not_frames():
    # при 4 кадр/с и 16 кадр/с вход засчитывается через одинаковое время
    times = []
    for fps in (4, 16):
        r = rule(min_inside_s=1.0)
        for i in range(100):
            ts = T0 + timedelta(seconds=i / fps)
            x = 300 if i < fps else 700                                # 1 с снаружи, потом внутри
            if r.update(RuleContext("cam-01", ts, W, H, [track(1, x)])):
                times.append((ts - T0).total_seconds())
                break
    assert times == [2.0, 2.0]


def test_reset_after_reconnect():
    r = rule(min_inside_s=0.5)
    assert len(flat(run(r, [{1: 700}] * 10))) == 1
    r.reset()                                                          # тот же track_id — уже другой человек
    assert len(flat(run(r, [{1: 700}] * 10, start=10))) == 1


def test_zone_of_other_camera_ignored():
    other = Zone("zone-9", "cam-02", ZONE.polygon)
    r = RULES["zone_intrusion"]("cam-01", {"zones": [other]})
    assert flat(run(r, [{1: 700}] * 10)) == []


def test_new_zone_version_resets_state():
    r = rule(min_inside_s=0.5)
    assert len(flat(run(r, [{1: 700}] * 10))) == 1
    r.set_zones([Zone("zone-1", "cam-01", ZONE.polygon, version=2)])
    ev = flat(run(r, [{1: 700}] * 10, start=10))
    assert len(ev) == 1 and ev[0].zone_version == 2


def test_zone_validation():
    with pytest.raises(ValueError):
        Zone("z", "cam-01", [(0, 0), (1, 1)])                         # 2 точки
    with pytest.raises(ValueError):
        Zone("z", "cam-01", [(0, 0), (1920, 0), (1920, 1080)])        # пиксели, а не доли


def test_zones_json_roundtrip_and_build_rules(tmp_path):
    from vision.analytics.zones import new_geometry_id

    path = tmp_path / "zones.json"
    zid = new_geometry_id("cam-01", "zone-1")
    save_zones(path, {zid: Zone(zid, "cam-01", ZONE.polygon, name="zone-1")})
    zones = load_zones(path)
    assert zones[zid].polygon == ZONE.polygon and zones[zid].name == "zone-1"
    rules = build_rules("cam-01", {"zone_intrusion": {"zones": ["zone-1"], "min_inside_s": 0.5}}, zones)  # по имени
    assert len(rules) == 1 and list(rules[0].zones) == [zid]
    assert build_rules("cam-01", {"zone_intrusion": {"zones": [zid]}}, zones)[0].zones.keys() == {zid}  # по UUID
    with pytest.raises(ValueError):
        build_rules("cam-01", {"zone_intrusion": {"zones": ["nope"]}}, zones)
    with pytest.raises(ValueError):
        build_rules("cam-01", {"no_such_rule": {}}, zones)


def test_zone_survives_pickle():
    import pickle
    ZONE.contains(0.7, 0.5)                                            # полигон уже построен
    z = pickle.loads(pickle.dumps(ZONE))
    assert z.contains(0.7, 0.5) and not z.contains(0.2, 0.5)


def test_zones_file_requires_uuid_and_migrates(tmp_path):
    import json

    from vision.analytics.zones import is_uuid, migrate_ids

    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": [ZONE.to_dict()]}), encoding="utf-8")       # старый id «zone-1»
    with pytest.raises(ValueError, match="UUID"):
        load_zones(path)
    (old, new), = migrate_ids(path)
    assert old == "zone-1" and is_uuid(new)
    z = load_zones(path)[new]
    assert z.name == "zone-1" and migrate_ids(path) == []                            # повторно — ничего
