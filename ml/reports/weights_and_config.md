# Веса и конфигурация узла ML — 2026-11-25 01:50

Артефакт ТЗ 5.3. Собрано `scripts/export_config_report.py` из `weights/`, `configs/` и окружения.

## Веса

| файл | что это | размер, МБ | sha256 | лицензия | источник |
|---|---|---|---|---|---|
| `fire_yolo26n.pt` | YOLO26n, дообучена на D-Fire (наша, Фаза 3.1) | — | **нет файла** — правило выключено | AGPL-3.0 (как у Ultralytics) | training/fire |
| `yolo26n-pose.pt` | Ultralytics YOLO26n-pose, COCO-Keypoints (готовые веса) | 7.5 | `eb3bb8268828aeaf…` | AGPL-3.0 | https://docs.ultralytics.com/tasks/pose |
| `yolo26n.pt` | Ultralytics YOLO26n, COCO (готовые веса) | 5.3 | `9b09cc8bf347f0fc…` | AGPL-3.0 | https://docs.ultralytics.com/models/yolo26/ |
| `buffalo_l/1k3d68.onnx` | InsightFace (не загружается) | 137.0 | `df5c06b8a0c12e42…` | только некоммерческие исследования | https://github.com/deepinsight/insightface |
| `buffalo_l/2d106det.onnx` | InsightFace (не загружается) | 4.8 | `f001b856447c4138…` | только некоммерческие исследования | https://github.com/deepinsight/insightface |
| `buffalo_l/det_10g.onnx` | InsightFace (используется) | 16.1 | `5838f7fe053675b1…` | только некоммерческие исследования | https://github.com/deepinsight/insightface |
| `buffalo_l/genderage.onnx` | InsightFace (не загружается) | 1.3 | `4fde69b1c810857b…` | только некоммерческие исследования | https://github.com/deepinsight/insightface |
| `buffalo_l/w600k_r50.onnx` | InsightFace (используется) | 166.3 | `4c06341c33c2ca1f…` | только некоммерческие исследования | https://github.com/deepinsight/insightface |

## Версии моделей в событиях (`model_version`)

```yaml
ml: 0.1.0
detector: yolo26n-det-1.0.0
pose: yolo26n-pose-1.0.0
tracker: bytetrack-1.0
fire: fire-yolo26n-0.0.0
rules:
  zone_intrusion: zone-0.1.0
  fall: fall-rule-0.1.0
  fight: fight-heur-0.1.0
  fire: fire-mofn-0.1.0
  smoke: smoke-mofn-0.1.0
  line_crossing: line-0.1.0
  loitering: loiter-0.1.0
  crowding: crowd-0.1.0
  running: run-0.1.0
  lying_motionless: lying-0.1.0
  camera_tampering: tamper-0.1.0
  abandoned_object: objects-0.1.0
  object_taken: objects-0.1.0
```

## Узел: `configs/app.yaml`

```yaml
detector:
  weights: weights/yolo26n.pt
  imgsz: 640
  conf: 0.1
  device: 0
  half: true
pose:
  mode: separate
  weights: weights/yolo26n-pose.pt
  imgsz: 640
  conf: 0.25
  device: 0
  half: true
  match_iou: 0.5
fire:
  weights: weights/fire_yolo26n.pt
  run_hz: 3
  imgsz: 640
  conf: 0.25
  device: 0
  half: true
tracker:
  type: bytetrack
  track_thresh: 0.5
  min_conf: 0.1
  min_hits: 3
  lost_s: 2.0
face:
  model: buffalo_l
  model_version: insightface-buffalo_l-1.0
  device: 0
  det_size: 320
  enroll_det_size: 640
  buffer_s: 5
  every_n: 3
  upper_frac: 0.5
  after_s: 2
  max_crops: 16
  min_quality: 0.4
  quality:
    min_det_score: 0.6
    min_side_px: 48
  storage: data/snapshots
node:
  node_id: ml-node-1
mqtt:
  enabled: true
  host: localhost
  port: 1883
  tls: false
  user_env: MQTT_USER
  keepalive_s: 10
  heartbeat_s: 5
  outbox: data/outbox.sqlite
  send_proposed: false
  max_age_s: 600
  dropped_log: data/outbox_dropped.jsonl
objects:
  classes:
    24: backpack
    26: handbag
    28: suitcase
    63: laptop
  tracker:
    track_thresh: 0.3
    min_hits: 3
    lost_s: 5.0
```

## Камеры и правила: `configs/cameras.yaml`

| камера | источник | process_fps | правило | параметры |
|---|---|---|---|---|
| cam-01 | `data/demo1.mp4` | 8 | zone_intrusion | `{"zones": ["zone-1"], "min_inside_s": 0.5, "min_outside_s": 1.0, "spawn_inside_counts": true}` |
| | | | fall | `{"still_s": 3.0}` |
| | | | fight | `{"scorer": "heuristic", "window_s": 2.0, "suspect_thr": 0.5, "confirm_thr": 0.7, "confirm_s": 1.0, "end_s": 3.0, "close_k": 1.0, "scorer_params": {"arm_speed": 1.5, "turn_speed": 0.8, "turn_deg": 90}}` |
| | | | fire | `{}` |
| | | | line_crossing | `{"lines": ["line-walk-1"], "direction": "any", "min_side_s": 0.5, "band": 0.03}` |
| | | | loitering | `{"zones": ["zone-1"], "dwell_s": 8, "gap_s": 3}` |
| | | | crowding | `{"zones": ["zone-hall"], "min_count": 8, "min_s": 3, "clear_s": 10}` |
| | | | running | `{"speed": 1.6, "min_s": 1.0}` |
| | | | lying_motionless | `{"lie_s": 30}` |
| | | | camera_tampering | `{"hold_s": 3}` |
| | | | abandoned_object | `{"abandon_s": 30, "attend_s": 5, "near_k": 1.0, "classes": ["backpack", "handbag", "suitcase", "laptop"]}` |
| | | | object_taken | `{"confirm_s": 2, "same_look": 0.35}` |
| cam-tamper | `data/tamper_test.mp4` | 8 | camera_tampering | `{}` |
| cam-fire | `data/fire1.mp4` | 8 | fire | `{"window_runs": 5, "min_hits": 4, "clear_s": 10, "classes": ["fire"], "min_conf": 0.25, "flicker": false, "flicker_min": 0.05}` |
| cam-nofire | `data/nofire1.mp4` | 8 | fire | `{}` |
| cam-fight | `data/fight1.mp4` | 8 | fight | `{}` |
| cam-fall | `data/fall1_rgb.avi` | 8 | fall | `{"fall_window_s": 1.0, "drop_ratio": 0.3, "torso_angle_deg": 60, "aspect_ratio": 1.0, "still_s": 3.0, "still_motion": 0.05, "get_up_s": 0.5, "kpt_conf": 0.5}` |
| cam-rtsp | `rtsp://localhost:8554/cam1` | 8 | — | `{}` |

## Зоны и линии: `configs/zones.json`

Часовой пояс расписаний: Asia/Almaty

| имя | id (UUID) | тип | камера | версия | активна | точек | расписание |
|---|---|---|---|---|---|---|---|
| zone-1 | `fbbdb037-4023-57e9-957c-75ce21f0822b` | зона | cam-01 | 1 | True | 4 | "" |
| zone-hall | `02354e71-a2d5-53c5-b706-4649b1be54fe` | зона | cam-01 | 1 | True | 4 | "" |
| zone-1-night | `5e832400-e004-55aa-9b46-1fd6178ad8c7` | зона | cam-01 | 1 | True | 4 | [{"days": "mon-fri", "from": "22:00", "to": "07:00"}, {"days": "sat,sun", "from": "00:00", "to": "24:00"}] |
| line-walk-1 | `c16ee1e7-5efb-58f0-a6af-adb38f62b8b0` | линия | cam-01 | 1 | True | 2 | "" |

## Окружение

- Python 3.13.5, Windows 11
- CUDA 13.2, GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- torch: 2.15.0+cu132
- torchvision: 0.29.2+cu132
- ultralytics: 8.4.214
- boxmot: 26.0.0
- opencv-python: 4.11.0.86
- insightface: 2.1
- onnxruntime-gpu: 1.32.0
- numpy: 2.5.4
- shapely: 2.2.0
- pydantic: 2.14.3
- paho-mqtt: 2.2.0
- pyyaml: 6.0.3
