# Артефакты ML-модуля (ТЗ 5.3)

| Артефакт ТЗ 5.3 | Где | Как пересобрать |
|---|---|---|
| Веса и конфигурация | [`weights_and_config.md`](weights_and_config.md) | `python scripts/export_config_report.py` |
| Отчёт об обучении (для базы — о данных и метриках) | [`data_report.md`](data_report.md) — данные, split, лицензии; [`eval/BASELINE.md`](eval/BASELINE.md) — метрики базы и разбор ошибок; [`eval/PACK_A.md`](eval/PACK_A.md) — новые правила (шаг 11) и их ложные в час; [`eval/PACK_B.md`](eval/PACK_B.md) — оставленный предмет и «забрал другой»; [`eval/PACK_D.md`](eval/PACK_D.md) — изменение охраняемого объекта; `eval/*.md|json` — отдельные прогоны | `python scripts/make_splits.py`, `python scripts/eval_events.py --task …` |
| Отчёт по скорости | [`bench/SUMMARY.md`](bench/SUMMARY.md), `bench/*.csv` | `python scripts/load_test.py` (на 4060 Laptop и 3080) |
| Примеры событий | [`../examples/events/`](../examples/events/) — JSON Schema, валидные и невалидные сообщения | `python scripts/export_contract.py` |

Дополнительно: [`faces/lfw_calibration.json`](faces/lfw_calibration.json) — проверка сервиса лиц на LFW
(не рабочий порог: он подбирается на добровольцах, M6).

Правило ТЗ 14.2: здесь только **измеренное**. Предполагаемая точность как достигнутая не пишется.
`eval/test_runs.csv` — журнал запусков на замороженном test (каждый запуск виден).
