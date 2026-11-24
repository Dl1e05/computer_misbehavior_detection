# Оценка: fight / dev — 2026-11-25 01:17

Роликов: 800 (все); process_fps 8.0; удержание последнего кадра 0.0 с; окно совпадения [начало − 1.0 с, конец + 2.0 с].
Правила: `{"running": {"speed": 1.6, "min_s": 1.0}, "lying_motionless": {"lie_s": 30}, "camera_tampering": {"hold_s": 3, "learn_s": 1}}`
Модели: `ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/run-0.1.0; ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/yolo26n-pose-1.0.0/lying-0.1.0; ml-0.1.0/tamper-0.1.0`

## running / detection

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

## lying_motionless / detection

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

## camera_tampering / detection

| precision | recall | F1 | TP | FN | FP | дубли | ложных в час | задержка p50 / p95, с |
|---|---|---|---|---|---|---|---|---|
| 0.0 | None | 0.0 | 0 | 0 | 3 | 0 | 2.7 (на 1.111 ч) | None / None |

Клипы (confusion matrix):

| | событие было | события не было |
|---|---|---|
| инцидент есть | 0 | 0 |
| инцидента нет | 3 | 797 |

По сценам:

| тег | роликов | precision | recall | FP | ложных в час |
|---|---|---|---|---|---|
| cctv | 800 | 0.0 | None | 3 | 2.7 |

Ошибки: пропусков 0, роликов с ложными 3 (полный список — в .json).

| ложное | теги | моменты, с | файл |
|---|---|---|---|
| rwf-fc9f6c2f8d | cctv | [4.5] | `data/raw/rwf2000/RWF-2000/RWF-2000/train/Train_NonFight/OTVE3eLAuUM_0.avi` |
| rwf-973931621b | cctv | [4.5] | `data/raw/rwf2000/RWF-2000/RWF-2000/train/Train_NonFight/sy7BoRGE_0.avi` |
| rwf-6039979e13 | cctv | [4.5] | `data/raw/rwf2000/RWF-2000/RWF-2000/train/Train_NonFight/uhTalZ00_0.avi` |
