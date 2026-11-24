# Оценка: fight / dev — 2026-11-25 01:49

Роликов: 800 (все); process_fps 8.0; удержание последнего кадра 0.0 с; окно совпадения [начало − 1.0 с, конец + 2.0 с].
Правила: `{"abandoned_object": {"abandon_s": 3, "attend_s": 5, "near_k": 1.0, "classes": ["backpack", "handbag", "suitcase", "laptop"]}, "object_taken": {"confirm_s": 2, "same_look": 0.35}}`
Модели: `ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/objects-0.1.0; ml-0.1.0/yolo26n-det-1.0.0/bytetrack-1.0/objects-0.1.0`

## abandoned_object / detection

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

## object_taken / detection

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
