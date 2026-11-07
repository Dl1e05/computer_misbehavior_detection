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
   - Ключи подписи (`SESSION_SIGNING_KEY`, `CLIP_LINK_SIGNING_KEY`) — минимум 32 символа и разные. Сгенерировать:
     ```bash
     uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
     ```
   - В `.env` только ключи приложения: неизвестный ключ остановит запуск (защита от опечаток). Настройки только для Docker Compose пишите в `compose.dev.yaml`.
3. Создать файл паролей Mosquitto (он в `.gitignore`, у каждого свой). Пароль берётся из `.env`; скобки `( … )` нужны, чтобы значения из `.env` не остались в терминале:
   ```bash
   : > mosquitto/passwd
   ( set -a; source .env; set +a
     docker run --rm -v "$PWD/mosquitto:/m" eclipse-mosquitto:2 \
       mosquitto_passwd -b /m/passwd "$MQTT_USERNAME" "$MQTT_PASSWORD" )
   sudo chown 1883:1883 mosquitto/passwd
   ```
4. Запустить PostgreSQL и MQTT:
   ```bash
   docker compose -f compose.dev.yaml up -d
   docker compose -f compose.dev.yaml ps   # database: healthy
   ```
   После изменения портов или паролей в `.env` пересоздайте контейнеры: `up -d --force-recreate`.
5. Запустить приложение: `uv run fastapi dev src/app/main.py`

## Проверки перед коммитом

```bash
uv run ruff format && uv run ruff check && uv run mypy && uv run pytest
```
