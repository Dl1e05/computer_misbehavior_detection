# Backend

API часть проекта computermisbehaviordetection.

Сервер, который принимает данные о поведении компьютера, передает их в ML-модель из `ml` и отдает результаты детекции фронтенду через API.

## Локальный запуск (dev)

Нужны Docker и [uv](https://docs.astral.sh/uv/).

1. Установить зависимости:
   ```bash
   uv sync
   ```
2. Создать `.env` из шаблона и заполнить **все** пустые значения:
   ```bash
   cp .env.example .env
   ```
   - `DB_PORT` — свободный порт (например `5433`, если на машине уже есть PostgreSQL на `5432`).
   - Два пользователя БД, с **разными** именами и паролями:
     - `DB_MIGRATION_USER` / `DB_MIGRATION_PASSWORD` — владелец схемы, только для миграций (Alembic);
     - `DB_USER` / `DB_PASSWORD` — ограниченный пользователь приложения: не может менять или удалять Audit (BE-FR-75). Его создаёт `postgres/init/` при первом запуске базы.
   - Ключи подписи (`SESSION_SIGNING_KEY`, `CLIP_LINK_SIGNING_KEY`) — минимум 32 символа и разные. Сгенерировать:
     ```bash
     uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
     ```
   - В `.env` только ключи приложения: неизвестный ключ остановит запуск (защита от опечаток). Настройки только для Docker Compose пишите в `compose.dev.yaml`.
3. Создать файл паролей Mosquitto (он в `.gitignore`, у каждого свой). Пароль берётся из `.env`; скобки `( … )` нужны, чтобы значения из `.env` не остались в терминале:
   ```bash
   sudo rm -f mosquitto/passwd
   ( set -a; source .env; set +a
     docker run --rm -v "$PWD/mosquitto:/m" eclipse-mosquitto:2 \
       mosquitto_passwd -c -b /m/passwd "$MQTT_USERNAME" "$MQTT_PASSWORD" )
   sudo chown 1883:1883 mosquitto/passwd
   sudo chmod 0600 mosquitto/passwd
   ```
   Шаг можно повторять (например, после смены `MQTT_PASSWORD`); затем перезапустите брокер: `docker compose -f compose.dev.yaml restart mqtt`.
4. Запустить PostgreSQL и MQTT:
   ```bash
   docker compose -f compose.dev.yaml up -d --wait   # ждёт, пока база станет healthy
   ```
   После изменения портов в `.env` пересоздайте контейнеры: `up -d --force-recreate`.
   Пользователи и пароли БД задаются **только при создании тома**: после их смены в `.env` нужно `docker compose -f compose.dev.yaml down -v` (удаляет данные dev-базы) и снова `up -d --wait`.
   Без `--wait` миграции могут запуститься, пока база ещё инициализируется (`Connection reset by peer`).
5. Применить миграции: `uv run alembic upgrade head`
6. Запустить приложение: `uv run fastapi dev src/app/main.py`
7. Запустить обработчик событий ML (отдельный процесс, только один экземпляр): `uv run python -m app.workers.events`
   Запустить таймеры (T4 через 10 с, эскалация через 30 с; можно несколько экземпляров): `uv run python -m app.workers.timers`
8. Отправить тестовое событие, как это делает ML (симулятор, только для разработки): `uv run python -m app.dev.publish_event fall`. Все варианты: `uv run python -m app.dev.publish_event --help`. Камера `cam-01` должна существовать в базе, иначе событие уйдёт в dead letters (`UNKNOWN_CAMERA`).

## Проверки

### Pre-commit (при каждом коммите)

Один раз на машину, из **корня репозитория** (там лежит `.pre-commit-config.yaml`):

```bash
uv tool install pre-commit
pre-commit install
```

При коммите, если среди изменений есть файлы из `backend/`, запускаются `ruff format`, `ruff check --fix`, `mypy` и быстрые тесты (`tests/domain`, без Docker). Если ruff что-то исправил, коммит останавливается: проверьте изменения, сделайте `git add` и закоммитьте ещё раз.

Запустить все проверки вручную: `pre-commit run --all-files`.

### Перед push

Полный набор тестов (нужен запущенный Docker):

```bash
uv run pytest
```
