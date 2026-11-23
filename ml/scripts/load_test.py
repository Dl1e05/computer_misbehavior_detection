"""Шаг 9: нагрузочный тест узла — N виртуальных камер, пока GPU/CPU тянут.

    python scripts/load_test.py                                  # N = 1,4,8,12,16,24,32; режимы persons и pose
    python scripts/load_test.py --cams 4 --modes pose            # «bench»: 4 камеры (T14)
    python scripts/load_test.py --cams 1,4,8 --duration 20 --imgsz 512

Каждая конфигурация — отдельный процесс (чистая видеопамять, падение по OOM не роняет тест):
N копий ролика как живые камеры (реальный темп, loop, process_fps), тот же Pipeline, что в демо,
лица и MQTT выключены (они срабатывают только на события и не нагружают поток кадров).

Режимы:
  persons — детектор + ByteTrack (минимум любой камеры: зона, будущие правила по трекам);
  pose    — + yolo26n-pose и правила падения и драки (самая тяжёлая камера базы).

Что пишем (reports/bench/<gpu>_<дата>.csv + .md):
  FPS на камеру (мин/сред) и суммарный, доля пропущенных кадров, задержка кадра от чтения до
  конца правил p50/p95 и её рост (p95 последней трети / первой), мс на батч по стадиям,
  средний батч, VRAM (torch peak + nvidia-smi), загрузка GPU и CPU, мощность GPU, режим питания.
«Тянет», если: каждая камера держит >= 95% process_fps, пропусков <= 5%, p95 задержки < 1 с
и не растёт (рост <= 1.5). Первое N, где это ломается, — предел узла.
Расчёт на 100 камер: узлов = ceil(100 / (предел × 0.7)).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODES = {
    "persons": {},
    "pose": {"fall": {}, "fight": {}},
}
COLUMNS = ["mode", "cams", "process_fps", "imgsz", "half", "duration_s", "ok", "why_not",
           "fps_cam_min", "fps_cam_mean", "fps_total", "drop_pct",
           "lat_p50_ms", "lat_p95_ms", "lat_growth",
           "batch_mean", "det_ms_p50", "det_ms_p95", "pose_ms_p50", "pose_ms_p95",
           "analytics_ms_p50", "analytics_ms_p95",
           "vram_torch_mb", "vram_smi_mb", "gpu_util_pct", "gpu_power_w", "cpu_pct",
           "gpu", "driver", "torch", "ultralytics", "cpu", "power_plugged", "source"]


# ---------------- окружение ----------------

def smi(query: str) -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return [x.strip() for x in out[0].split(",")] if out else []
    except (OSError, subprocess.SubprocessError):
        return []


def environment() -> dict:
    import torch
    import ultralytics
    import psutil

    gpu, driver = (smi("name,driver_version") + ["?", "?"])[:2]
    batt = psutil.sensors_battery()
    return {"gpu": gpu, "driver": driver, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "cpu": f"{platform.processor() or platform.machine()} ({psutil.cpu_count()} потоков)",
            "power_plugged": "n/a" if batt is None else ("сеть" if batt.power_plugged else "батарея")}


class Sampler(threading.Thread):
    """Раз в секунду: загрузка GPU, VRAM, мощность (nvidia-smi) и CPU (psutil)."""

    def __init__(self):
        super().__init__(daemon=True)
        import psutil

        self.psutil = psutil
        self.rows: list[tuple[float, float, float, float]] = []
        self.stop = threading.Event()
        psutil.cpu_percent(None)

    def run(self):
        while not self.stop.wait(1.0):
            v = smi("utilization.gpu,memory.used,power.draw")
            try:
                util, mem, power = (float(x) for x in v[:3])
            except (ValueError, TypeError):
                util = mem = power = float("nan")
            self.rows.append((util, mem, power, self.psutil.cpu_percent(None)))

    def summary(self) -> dict:
        if not self.rows:
            return {}
        a = np.array(self.rows)
        return {"gpu_util_pct": round(float(np.nanmean(a[:, 0])), 1), "vram_smi_mb": round(float(np.nanmax(a[:, 1]))),
                "gpu_power_w": round(float(np.nanmean(a[:, 2])), 1), "cpu_pct": round(float(np.nanmean(a[:, 3])), 1)}


def pct(values, q):
    return round(float(np.percentile(values, q)), 1) if len(values) else None


# ---------------- одна конфигурация (дочерний процесс) ----------------

def worker(cfg: dict) -> dict:
    import logging

    import torch
    import yaml

    from vision.runtime.bus import FrameBus
    from vision.runtime.pipeline import Pipeline

    logging.basicConfig(level=logging.WARNING)
    app = yaml.safe_load((ROOT / "configs/app.yaml").read_text(encoding="utf-8"))
    for stage in ("detector", "pose"):
        app[stage]["imgsz"] = cfg["imgsz"]
        app[stage]["half"] = cfg["half"]
    cams = [{"id": f"cam-{i + 1:02d}", "source": cfg["source"], "process_fps": cfg["fps"], "realtime": True,
             "loop": True, "rules": MODES[cfg["mode"]]} for i in range(cfg["cams"])]
    node = Pipeline(app, cams, {}, ROOT, faces=False, mqtt=False)
    bus = FrameBus(node.sources, drop_old=True).start()

    t_end_warm = time.monotonic() + cfg["warmup"]
    while time.monotonic() < t_end_warm:                       # разгон: декодеры, кэши cudnn, батчи
        node.process(bus.next_batch(timeout=0.5))
    node.metrics.reset()
    torch.cuda.reset_peak_memory_stats()
    prod0, drop0 = dict(bus.produced), dict(bus.dropped)
    sampler = Sampler()
    sampler.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < cfg["duration"]:
        node.process(bus.next_batch(timeout=0.5))
    dur = time.monotonic() - t0
    sampler.stop.set()
    bus.stop()
    node.close()

    m = node.metrics
    fps_cam = [m.frames.get(c["id"], 0) / dur for c in cams]
    produced = sum(bus.produced[c] - prod0[c] for c in prod0)
    dropped = sum(bus.dropped[c] - drop0[c] for c in drop0)
    lat = [x[1] for x in m.latency_ms]
    third = max(len(lat) // 3, 1)
    growth = (np.percentile(lat[-third:], 95) / max(np.percentile(lat[:third], 95), 1e-6)) if len(lat) >= 9 else 1.0
    res = {
        "mode": cfg["mode"], "cams": cfg["cams"], "process_fps": cfg["fps"], "imgsz": cfg["imgsz"],
        "half": cfg["half"], "duration_s": round(dur, 1),
        "fps_cam_min": round(min(fps_cam), 2), "fps_cam_mean": round(float(np.mean(fps_cam)), 2),
        "fps_total": round(sum(fps_cam), 1), "drop_pct": round(100 * dropped / max(produced, 1), 1),
        "lat_p50_ms": pct(lat, 50), "lat_p95_ms": pct(lat, 95), "lat_growth": round(float(growth), 2),
        "batch_mean": round(float(np.mean(m.batch_size)), 2) if m.batch_size else 0,
        "det_ms_p50": pct(m.det_ms, 50), "det_ms_p95": pct(m.det_ms, 95),
        "pose_ms_p50": pct(m.pose_ms, 50), "pose_ms_p95": pct(m.pose_ms, 95),
        "analytics_ms_p50": pct(m.analytics_ms, 50), "analytics_ms_p95": pct(m.analytics_ms, 95),
        "vram_torch_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "source": Path(cfg["source"]).name,
        **sampler.summary(),
    }
    why = []
    if res["fps_cam_min"] < 0.95 * cfg["fps"]:
        why.append(f"fps {res['fps_cam_min']} < {0.95 * cfg['fps']:.1f}")
    if res["drop_pct"] > 5:
        why.append(f"пропуски {res['drop_pct']}%")
    if (res["lat_p95_ms"] or 0) > 1000:
        why.append(f"p95 {res['lat_p95_ms']} мс")
    if res["lat_growth"] > 1.5:
        why.append(f"задержка растёт ×{res['lat_growth']}")
    res["ok"] = not why
    res["why_not"] = "; ".join(why)
    return res


# ---------------- оркестратор ----------------

def run_config(cfg: dict) -> dict:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([sys.executable, __file__, "--worker", json.dumps(cfg)], capture_output=True, text=True,
                       encoding="utf-8", env=env, timeout=cfg["duration"] + cfg["warmup"] + 300)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
    return {**{k: cfg.get(k) for k in ("mode", "cams", "imgsz", "half")}, "process_fps": cfg["fps"],
            "ok": False, "why_not": "упал: " + " | ".join(tail)[:300]}


def report_md(rows: list[dict], env: dict, out: Path, fps: float) -> str:
    lines = [f"# Нагрузочный тест узла ML — {datetime.now():%Y-%m-%d %H:%M}", "",
             f"GPU: {env['gpu']} (драйвер {env['driver']}), CPU: {env['cpu']}, питание: {env['power_plugged']}",
             f"torch {env['torch']}, ultralytics {env['ultralytics']}; process_fps {fps}; CSV: `{out.name}`", "",
             "| режим | камер | тянет | FPS/камеру мин | пропуски % | задержка p50/p95 мс | рост | батч | "
             "детектор p95 мс | поза p95 мс | VRAM МБ | GPU % | CPU % |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['cams']} | {'да' if r['ok'] else 'нет: ' + r.get('why_not', '')} | "
                     f"{r.get('fps_cam_min')} | {r.get('drop_pct')} | {r.get('lat_p50_ms')}/{r.get('lat_p95_ms')} | "
                     f"{r.get('lat_growth')} | {r.get('batch_mean')} | {r.get('det_ms_p95')} | {r.get('pose_ms_p95')} | "
                     f"{r.get('vram_smi_mb')} | {r.get('gpu_util_pct')} | {r.get('cpu_pct')} |")
    lines += ["", "## Расчёт на 100 камер", "",
              "Предел узла — наибольшее проверенное N, где все камеры держат process_fps. "
              "Узлов = ceil(100 / (предел × 0.7)), 0.7 — запас.", ""]
    for mode in dict.fromkeys(r["mode"] for r in rows):
        ok = [r["cams"] for r in rows if r["mode"] == mode and r["ok"]]
        failed = [r["cams"] for r in rows if r["mode"] == mode and not r["ok"]]
        if not ok:
            lines.append(f"- **{mode}**: не тянет даже {min(failed)} камер(ы)")
            continue
        limit = max(ok)
        bound = f" (сломалось на {min(f for f in failed if f > limit)})" if any(f > limit for f in failed) else \
            " (верхняя граница не найдена — увеличь --cams)"
        nodes = math.ceil(100 / (limit * 0.7))
        lines.append(f"- **{mode}**: предел ≥ {limit} камер{bound} → на 100 камер нужно **{nodes}** таких узлов")
    lines += ["", "Все камеры одного режима; в жизни камеры разные (часть только с зоной, часть с позой) — "
              "считать по доле камер каждого режима. Числа ноутбучной GPU на десктопную не переносим (ТЗ 14.1)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--cams", default="1,4,8,12,16,24,32", help="список N через запятую")
    p.add_argument("--modes", default="persons,pose", help=f"из {list(MODES)}")
    p.add_argument("--source", default="data/demo1.mp4", help="ролик для всех виртуальных камер")
    p.add_argument("--fps", type=float, default=8.0, help="process_fps каждой камеры")
    p.add_argument("--duration", type=float, default=30.0, help="секунд замера на конфигурацию")
    p.add_argument("--warmup", type=float, default=8.0, help="секунд разгона перед замером")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--no-half", action="store_true", help="FP32 вместо FP16")
    p.add_argument("--keep-going", action="store_true", help="не останавливаться после первого «не тянет»")
    p.add_argument("--tag", default="", help="метка в имени отчёта, напр. trt или 3080")
    args = p.parse_args()

    if args.worker:
        print("RESULT " + json.dumps(worker(json.loads(args.worker)), ensure_ascii=False), flush=True)
        return

    env = environment()
    src = str((ROOT / args.source).resolve()) if not Path(args.source).is_absolute() else args.source
    slug = env["gpu"].replace("NVIDIA GeForce ", "").replace(" ", "-").lower()
    out = ROOT / "reports/bench" / f"{slug}_{datetime.now():%Y%m%d-%H%M}{'_' + args.tag if args.tag else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{env['gpu']} | {env['cpu']} | питание: {env['power_plugged']} -> {out}")
    rows = []
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for mode in args.modes.split(","):
            for n in (int(x) for x in args.cams.split(",")):
                cfg = {"mode": mode, "cams": n, "fps": args.fps, "imgsz": args.imgsz, "half": not args.no_half,
                       "duration": args.duration, "warmup": args.warmup, "source": src}
                t = time.monotonic()
                r = {**run_config(cfg), **env}
                rows.append(r)
                w.writerow(r)
                fh.flush()
                print(f"  {mode:8} N={n:3}: {'ТЯНЕТ ' if r['ok'] else 'НЕ ТЯНЕТ'} "
                      f"fps/кам мин {r.get('fps_cam_min')}, пропуски {r.get('drop_pct')}%, "
                      f"задержка p95 {r.get('lat_p95_ms')} мс (рост ×{r.get('lat_growth')}), "
                      f"батч {r.get('batch_mean')}, GPU {r.get('gpu_util_pct')}%, CPU {r.get('cpu_pct')}%, "
                      f"VRAM {r.get('vram_smi_mb')} МБ  [{time.monotonic() - t:.0f} с]"
                      + (f"  — {r['why_not']}" if not r["ok"] else ""), flush=True)
                if not r["ok"] and not args.keep_going:
                    break
    md = report_md(rows, env, out, args.fps)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
