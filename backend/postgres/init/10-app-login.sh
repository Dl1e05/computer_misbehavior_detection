#!/bin/sh
# Runs once, when the database volume is first created
# (docker-entrypoint-initdb.d). Creates the limited login the app connects
# with. Its rights come from the app_runtime role, which the migrations grant
# (BE-FR-75); the migration creates the role only if it does not exist yet.
set -eu

if [ "$APP_DB_USER" = "$POSTGRES_USER" ]; then
    echo "DB_USER must differ from DB_MIGRATION_USER in .env" >&2
    exit 1
fi
if [ "$APP_DB_USER" = "app_runtime" ]; then
    echo "DB_USER must not be app_runtime: that is the group role" >&2
    exit 1
fi
# A leaked app password must not also open the owner account.
if [ "$APP_DB_PASSWORD" = "$POSTGRES_PASSWORD" ]; then
    echo "DB_PASSWORD must differ from DB_MIGRATION_PASSWORD in .env" >&2
    exit 1
fi

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    --set app_user="$APP_DB_USER" --set app_password="$APP_DB_PASSWORD" <<'SQL'
CREATE ROLE app_runtime NOLOGIN;
CREATE ROLE :"app_user" LOGIN PASSWORD :'app_password' IN ROLE app_runtime;
SQL
