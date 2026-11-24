"""Артефакт ТЗ 5.3 «Веса и конфигурация»: какие веса (sha256, источник, лицензия), версии библиотек,
все пороги узла и правил по камерам — одним файлом reports/weights_and_config.md.

    python scripts/export_config_report.py

Пересобирать при любой смене весов или порогов (файл в git — видно, что поменялось).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision.sources.video_source import mask_password  # noqa: E402

WEIGHT_INFO = {
    "yolo26n.pt": ("Ultralytics YOLO26n, COCO (готовые веса)", "AGPL-3.0",
                   "https://docs.ultralytics.com/models/yolo26/"),
    "yolo26n-pose.pt": ("Ultralytics YOLO26n-pose, COCO-Keypoints (готовые веса)", "AGPL-3.0",
                        "https://docs.ultralytics.com/tasks/pose"),
    "fire_yolo26n.pt": ("YOLO26n, дообучена на D-Fire (наша, Фаза 3.1)", "AGPL-3.0 (как у Ultralytics)",
                        "training/fire"),
}
PACKAGES = ["torch", "torchvision", "ultralytics", "boxmot", "opencv-python", "insightface", "onnxruntime-gpu",
            "numpy", "shapely", "pydantic", "paho-mqtt", "pyyaml"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pkg(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "—"


def main() -> None:
    app = yaml.safe_load((ROOT / "configs/app.yaml").read_text(encoding="utf-8"))
    cams = yaml.safe_load((ROOT / "configs/cameras.yaml").read_text(encoding="utf-8")).get("cameras", [])
    zones_path = ROOT / "configs/zones.json"
    zones = json.loads(zones_path.read_text(encoding="utf-8")) if zones_path.exists() else []

    L = [f"# Веса и конфигурация узла ML — {datetime.now():%Y-%m-%d %H:%M}", "",
         "Артефакт ТЗ 5.3. Собрано `scripts/export_config_report.py` из `weights/`, `configs/` и окружения.", "",
         "## Веса", "", "| файл | что это | размер, МБ | sha256 | лицензия | источник |", "|---|---|---|---|---|---|"]
    wanted = {Path(app[s]["weights"]).name for s in ("detector", "pose", "fire") if s in app}
    for name in sorted(wanted | {p.name for p in (ROOT / "weights").glob("*.pt")}):
        what, lic, src = WEIGHT_INFO.get(name, ("—", "—", "—"))
        p = ROOT / "weights" / name
        if p.exists():
            L.append(f"| `{name}` | {what} | {p.stat().st_size / 2**20:.1f} | `{sha256(p)[:16]}…` | {lic} | {src} |")
        else:
            L.append(f"| `{name}` | {what} | — | **нет файла** — правило выключено | {lic} | {src} |")
    face_dir = Path.home() / ".insightface" / "models" / app.get("face", {}).get("model", "buffalo_l")
    for f in sorted(face_dir.glob("*.onnx")) if face_dir.exists() else []:
        used = f.name in ("det_10g.onnx", "w600k_r50.onnx")
        L.append(f"| `{face_dir.name}/{f.name}` | InsightFace {'(используется)' if used else '(не загружается)'} | "
                 f"{f.stat().st_size / 2**20:.1f} | `{sha256(f)[:16]}…` | только некоммерческие исследования | "
                 "https://github.com/deepinsight/insightface |")

    L += ["", "## Версии моделей в событиях (`model_version`)", "", "```yaml",
          yaml.safe_dump(app.get("versions", {}), allow_unicode=True, sort_keys=False).strip(), "```"]

    shown = {k: v for k, v in app.items() if k not in ("versions",)}
    if "mqtt" in shown:
        shown["mqtt"] = {k: v for k, v in shown["mqtt"].items() if "pass" not in k.lower()}
    L += ["", "## Узел: `configs/app.yaml`", "", "```yaml",
          yaml.safe_dump(shown, allow_unicode=True, sort_keys=False).strip(), "```"]

    L += ["", "## Камеры и правила: `configs/cameras.yaml`", "",
          "| камера | источник | process_fps | правило | параметры |", "|---|---|---|---|---|"]
    for c in cams:
        rules = c.get("rules") or {"—": {}}
        for i, (rule, params) in enumerate(rules.items()):
            head = (f"| {c['id']} | `{mask_password(str(c['source']))}` | {c.get('process_fps', 8)} "
                    if i == 0 else "| | | ")
            L.append(f"{head}| {rule} | `{json.dumps(params or {}, ensure_ascii=False)}` |")

    L += ["", "## Зоны и линии: `configs/zones.json`", "",
          f"Часовой пояс расписаний: {zones.get('timezone', 'пояс машины')}", "",
          "| имя | id (UUID) | тип | камера | версия | активна | точек | расписание |", "|---|---|---|---|---|---|---|---|"]
    for kind, key, pts in (("зона", "zone_id", "polygon"), ("линия", "line_id", "points")):
        for z in zones.get("zones" if kind == "зона" else "lines", []):
            L.append(f"| {z.get('name', '')} | `{z.get(key)}` | {kind} | {z.get('camera_id')} | {z.get('version')} | "
                     f"{z.get('active', True)} | {len(z.get(pts, []))} | {json.dumps(z.get('schedule', ''), ensure_ascii=False)} |")

    L += ["", "## Окружение", "", f"- Python {platform.python_version()}, {platform.system()} {platform.release()}"]
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет CUDA"
        L.append(f"- CUDA {torch.version.cuda}, GPU: {gpu}")
    except Exception:  # noqa: BLE001
        pass
    L += [f"- {n}: {pkg(n)}" for n in PACKAGES]
    out = ROOT / "reports" / "weights_and_config.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
