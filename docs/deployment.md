# Deployment

Two supported ways to run the platform: **Docker Compose** (recommended) and
**bare metal** (what the development docs describe).

---

## 1. Docker Compose

### Prerequisites

- Docker Engine 24+ with Compose v2
- ~6GB free disk (the backend image is large — see [Image size](#image-size))
- Internet access **at build time** (model weights are baked into the image;
  at run time the container needs no outbound network)

### Steps

```bash
cp .env.docker.example .env
```

Fill in the two secrets — the backend **refuses to start** in production
without them (`app/core/security.py::validate_production_settings`):

```bash
python -c "import secrets; print('API_KEY=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
```

Then:

```bash
docker compose up --build
```

Open <http://localhost:8080>. The first build takes 10–20 minutes, almost all
of it downloading PyTorch and the ViT weights.

### What runs where

| Service    | Image base          | Port                | Purpose                                    |
|------------|---------------------|---------------------|--------------------------------------------|
| `frontend` | `nginx:1.27-alpine` | `8080` → 80         | Serves the SPA, proxies `/api/` to backend |
| `backend`  | `python:3.14-slim`  | internal only, 8000 | FastAPI + inference pipeline               |

**The backend port is not published to the host.** Everything enters through
nginx. That is deliberate: the API-key check and the rate limiter are only
meaningful if there is no second route in. Uncomment the `ports` block in
`docker-compose.yml` if you need to curl the API directly during debugging,
and comment it back out afterwards.

### Persistent data

Three named volumes: `uploads`, `reports`, `dbdata`. They survive
`docker compose down`; `docker compose down -v` **deletes them**, including
every stored analysis.

Named volumes rather than bind mounts, because the container runs as uid
10001 and a bind-mounted host directory typically arrives root-owned — which
fails at the first write with a permission error that reads like an
application bug.

Back up the database with:

```bash
docker compose exec backend sh -c "sqlite3 /data/db/app.db .dump" > backup.sql
```

---

## 2. Bare metal

See [`README.md`](../README.md) for the development setup. For a production
run on a host:

```bash
cd backend && alembic upgrade head
```

```bash
APP_ENV=production API_KEY=... SECRET_KEY=... uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Then build and serve the frontend from any static host:

```bash
cd frontend && VITE_API_BASE_URL=https://your-host/api/v1 npm run build
```

**Keep `--workers 1`.** Each uvicorn worker is a separate process with its own
copy of the ViT (~330MB resident) *and* its own in-memory rate-limiter state,
so N workers means N× the memory and an N× looser effective rate limit.
Scale with the in-process thread pool (`MAX_CONCURRENT_ANALYSES`) instead —
inference releases the GIL, so this genuinely parallelises.

---

## Configuration

Every setting in `app/core/config.py` is overridable by an environment
variable of the same name, upper-cased. The ones that matter in production:

| Variable                  | Default          | Notes                                                                |
|---------------------------|------------------|----------------------------------------------------------------------|
| `APP_ENV`                 | `development`    | Anything other than `development`/`test` enables the production gate  |
| `API_KEY`                 | *(empty)*        | **Required** in production. Sent as `X-API-Key`                       |
| `SECRET_KEY`              | `change-me-...`  | **Required** in production                                            |
| `CORS_ORIGINS`            | localhost:5173   | Must match the address users actually type, including port            |
| `TRUST_PROXY_HEADERS`     | `false`          | Only `true` when a trusted proxy is the *sole* ingress                |
| `MAX_CONCURRENT_ANALYSES` | `2`              | More than physical cores just thrashes                                |
| `MAX_UPLOAD_SIZE_MB`      | `200`            | Raise `client_max_body_size` in `frontend/default.conf.template` to match                 |
| `DATABASE_URL`            | local SQLite     | See [Scaling limits](#scaling-limits)                                 |

`APP_ENV=production` with a missing or default secret is a **startup failure**,
not a warning. An unauthenticated deployment cannot happen by forgetting a
variable.

---

## Things that will bite you

### `libGL.so.1: cannot open shared object file`

`requirements.txt` pins `opencv-python-headless` specifically to avoid GUI
bindings — but **MediaPipe declares a hard dependency on
`opencv-contrib-python`** (the non-headless build), so pip installs it anyway
and `import cv2` resolves to a binary linked against libGL. The backend
Dockerfile installs `libgl1` and `libglib2.0-0` for exactly this reason. Drop
them from a "minimal" image and it builds cleanly, then dies on first import.

### Image size

The backend image is large: PyTorch CPU, the ViT weights (~330MB), MediaPipe,
OpenCV, and the Python base. The Dockerfile already installs torch from
`download.pytorch.org/whl/cpu` — the default PyPI wheel bundles CUDA libraries
worth roughly another 2GB that would never execute, since the pipeline is
CPU-only by design.

Further reduction would mean a multi-stage build copying only site-packages
into a distroless base. Not done: it complicates the build for a saving that
does not change what the service can do.

### The browser never holds the API key

The SPA does not send `X-API-Key`, and it must not: Vite inlines env vars into
the bundle at build time, so a `VITE_API_KEY` would be readable by anyone who
opens devtools. Instead **nginx injects the key** into proxied requests
(`proxy_set_header X-API-Key`), rendered at container start from `SDP_API_KEY`
via the nginx image's template mechanism — so it stays out of both the bundle
and `docker history`.

The consequence, stated plainly: **anyone who can reach nginx can use the API.**
The shared key authenticates the *client* (here, the proxy), not a user — which
is what `README.md` already lists as an open gap. What it still buys is that the
backend rejects traffic that bypasses the proxy. If you expose this beyond a
trusted network, put real per-user auth in front of it.

### Cold start

The first request after a container start loads the ViT into memory and takes
noticeably longer than subsequent ones. The compose healthcheck uses a 90s
`start_period` so this does not restart-loop a service that is merely warming
up.

### Migrations run in the entrypoint, not at app startup

`docker-entrypoint.sh` runs `alembic upgrade head` before exec'ing uvicorn.
With more than one replica, N processes racing to `ALTER` the same table is a
real corruption risk, and a failed migration should stop the container loudly
rather than leave a half-migrated schema serving traffic.

---

## Scaling limits

Two hard limits, both from SQLite:

1. **One writer at a time.** WAL mode allows concurrent readers, but writes
   serialise. Fine for the expected load (analyses take seconds and are far
   rarer than reads); not fine at high write concurrency.
2. **The database is a file on a volume**, so replicas cannot share it across
   hosts.

Past a single host, the migration path is: point `DATABASE_URL` at PostgreSQL
(SQLAlchemy already abstracts this, and the Alembic migrations were written in
batch mode, which Postgres also accepts), move uploads to object storage, and
replace the in-process thread pool with a real queue (Celery/RQ + Redis). None
of that is needed at the scale this project targets, and building it now would
be speculative infrastructure.

The in-memory rate limiter is per-process, so it also stops being correct
under multiple replicas — it would need to move to Redis at the same time.

---

## Health and troubleshooting

```bash
docker compose ps
```

```bash
docker compose logs -f backend
```

| Symptom                            | Cause                                                                                  |
|------------------------------------|----------------------------------------------------------------------------------------|
| Backend exits immediately          | Missing `API_KEY`/`SECRET_KEY` under `APP_ENV=production`                               |
| `413 Request Entity Too Large`     | `client_max_body_size` in `frontend/default.conf.template` below `MAX_UPLOAD_SIZE_MB`                       |
| 504 on a long video                | `proxy_read_timeout` too low for the analysis duration                                  |
| Analyses stuck in `processing`     | Container restarted mid-flight; `recover_stale_records()` fails them on the next start  |
| 401 on every request               | Client not sending `X-API-Key`                                                          |
| Frontend loads, all API calls fail | Built with an absolute `VITE_API_BASE_URL`; rebuild with `/api/v1`                      |
