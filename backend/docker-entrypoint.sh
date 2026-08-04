#!/bin/sh
# Applies database migrations, then hands off to the CMD.
#
# Migrations run HERE rather than in application startup on purpose: with more
# than one replica, N processes racing to ALTER the same table is a genuine
# corruption risk, and a migration failure should stop the container loudly
# instead of leaving a half-migrated schema serving traffic.
set -e

cd /app/backend

echo "[entrypoint] applying database migrations..."
alembic upgrade head
echo "[entrypoint] migrations applied."

exec "$@"
