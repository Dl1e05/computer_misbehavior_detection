# Оценка: fight / dev — 2026-11-25 02:44

Роликов: 800 (все); process_fps 8.0; удержание последнего кадра 0.0 с; окно совпадения [начало − 1.0 с, конец + 2.0 с].
Правила: `{"fall": {"fall_window_s": 1.0, "drop_ratio": 0.3, "torso_angle_deg": 60, "aspect_ratio": 1.0, "still_s": 3.0, "still_motion": 0.05, "get_up_s": 0.5, "kpt_conf": 0.5, "bbox_fallback": false}}`
Модели: `ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/yolo26n-pose-1.0.0/fall-rule-0.1.0`

## fall / detection

| precision | recall | F1 | TP | FN | FP | дубли | ложных в час | задержка p50 / p95, с |
|---|---|---|---|---|---|---|---|---|
| None | None | None | 0 | 0 | 0 | 0 | 0.0 (на 1.111 ч) | None / None |

Клипы (confusion matrix):

| | событие было | события не было |
|---|---|---|
| инцидент есть | 0 | 0 |
| инцидента нет | 0 | 800 |

По сценам:

| тег | роликов | precision | recall | FP | ложных в час |
|---|---|---|---|---|---|
| cctv | 800 | None | None | 0 | 0.0 |

Ошибки: пропусков 0, роликов с ложными 0 (полный список — в .json).
