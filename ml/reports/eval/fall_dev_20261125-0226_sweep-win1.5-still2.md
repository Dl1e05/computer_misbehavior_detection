# Оценка: fall / dev — 2026-11-25 02:26

Роликов: 57 (все); process_fps 8.0; удержание последнего кадра 5.0 с; окно совпадения [начало − 1.0 с, конец + 2.0 с].
Правила: `{"fall": {"fall_window_s": 1.5, "drop_ratio": 0.3, "torso_angle_deg": 60, "aspect_ratio": 1.0, "still_s": 2, "still_motion": 0.05, "get_up_s": 0.5, "kpt_conf": 0.5}}`
Модели: `ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/yolo26n-pose-1.0.0/fall-rule-0.1.0`

## fall / detection

| precision | recall | F1 | TP | FN | FP | дубли | ложных в час | задержка p50 / p95, с |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.267 | 0.421 | 8 | 22 | 0 | 0 | 0.0 (на 0.108 ч) | 3.9 / 4.386 |

Клипы (confusion matrix):

| | событие было | события не было |
|---|---|---|
| инцидент есть | 8 | 22 |
| инцидента нет | 0 | 27 |

По сценам:

| тег | роликов | precision | recall | FP | ложных в час |
|---|---|---|---|---|---|
| adl | 27 | None | None | 0 | 0.0 |
| adl_lying | 12 | None | None | 0 | 0.0 |
| overhead | 15 | None | 0.0 | 0 | 0.0 |
| side | 42 | 1.0 | 0.533 | 0 | 0.0 |

Ошибки: пропусков 22, роликов с ложными 0 (полный список — в .json).

| пропуск | теги | файл |
|---|---|---|
| fall-01-cam1 | overhead | `data/raw/urfall/fall-01-cam1.mp4` |
| fall-02-cam1 | overhead | `data/raw/urfall/fall-02-cam1.mp4` |
| fall-03-cam0 | side | `data/raw/urfall/fall-03-cam0.mp4` |
| fall-03-cam1 | overhead | `data/raw/urfall/fall-03-cam1.mp4` |
| fall-05-cam0 | side | `data/raw/urfall/fall-05-cam0.mp4` |
| fall-05-cam1 | overhead | `data/raw/urfall/fall-05-cam1.mp4` |
| fall-07-cam1 | overhead | `data/raw/urfall/fall-07-cam1.mp4` |
| fall-09-cam1 | overhead | `data/raw/urfall/fall-09-cam1.mp4` |
| fall-14-cam1 | overhead | `data/raw/urfall/fall-14-cam1.mp4` |
| fall-15-cam1 | overhead | `data/raw/urfall/fall-15-cam1.mp4` |
| fall-16-cam1 | overhead | `data/raw/urfall/fall-16-cam1.mp4` |
| fall-19-cam0 | side | `data/raw/urfall/fall-19-cam0.mp4` |
| fall-19-cam1 | overhead | `data/raw/urfall/fall-19-cam1.mp4` |
| fall-21-cam0 | side | `data/raw/urfall/fall-21-cam0.mp4` |
| fall-21-cam1 | overhead | `data/raw/urfall/fall-21-cam1.mp4` |
