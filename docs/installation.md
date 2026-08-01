# Installation Guide

> Grows with each milestone. Currently the backend (Milestone 2) and frontend (Milestone 3)
> skeletons are runnable end to end; the AI pipeline, uploads, and history are not built yet.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Git

## 1. Clone & configure

```bash
git clone <repo-url>
cd sports-deepfake-platform
cp .env.example .env
```

## 2. Backend

> **Note (Milestone 9):** the `Analysis` table gained a new column
> (`detector_breakdown`). Tables are created via `Base.metadata.create_all()`, which only
> creates *missing* tables — it won't add columns to one that already exists. If you have a
> `database/app.db` from before Milestone 9, delete it (`rm database/app.db`) and restart the
> backend to let it recreate with the current schema.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env           # optional — defaults already work for local dev
uvicorn app.main:app --reload
```

Windows users can also run `scripts/run_backend.ps1` (equivalent of `scripts/run_backend.sh`,
which requires a POSIX shell / WSL).

For local development (tests, linting, type-checking), install
`requirements-dev.txt` instead — it pulls in `requirements.txt` plus `pytest`, `ruff`, `mypy`:

```bash
pip install -r requirements-dev.txt
```

> **Note (Milestone 7+):** `requirements.txt` now includes `torch`, `torchvision`, and
> `transformers` (~1.5GB combined) for the deep learning detector. The first time detection
> code actually runs, it downloads a ~330MB pretrained model from Hugging Face Hub — this
> requires normal outbound internet access (works fine on WSL). See `docs/models.md` for details.

> **Note (R1, Python 3.14):** `numpy`, `pydantic`/`pydantic-settings`, and `SQLAlchemy` are
> pinned to versions with a Python 3.14 wheel (bumped from their original pins — see comments in
> `requirements.txt` for exact versions and why). Below 3.14 the original pins should still work,
> but this has only been verified against 3.14 so far.
>
> **Known packaging quirk:** `mediapipe` pulls in `opencv-contrib-python` (unpinned) as a
> transitive dependency, which installs into the same `cv2/` namespace as this project's own
> `opencv-python-headless` pin. Whichever installs last wins the file collision — on the
> Windows/Python 3.14 install this was verified against, that's `opencv-contrib-python`, so
> `cv2.__version__` reports `5.0.0` rather than the pinned `4.10.0.84`. This has been confirmed
> harmless for every `cv2` function this codebase actually calls (contrib is a superset), but it
> is a real reproducibility gap worth knowing about — and `opencv-contrib-python` (unlike the
> `-headless` variant) pulls in GUI bindings that may need extra system libraries in a minimal
> Docker image (relevant for the Milestone 16/R14 deployment work).

- API root: http://localhost:8000/
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

Run tests:
```bash
pytest
```

## 3. Frontend

In a separate terminal:

```bash
cd frontend
npm install
cp .env.example .env           # optional — defaults already point at localhost:8000
npm run dev
```

- App: http://localhost:5173/

Open the app — the **Dashboard** page shows a live "Backend Connection" card. It should read
**Online** with the app name/version/environment if both servers are running and reachable.

## 4. Verifying the full stack

1. Backend running on port 8000 (`uvicorn app.main:app --reload`)
2. Frontend running on port 5173 (`npm run dev`)
3. Visit http://localhost:5173/ → Dashboard → "Backend Connection" shows **Online**

If it shows **Offline**, check that the backend is running and that
`VITE_API_BASE_URL` (frontend `.env`) matches where the backend is actually listening.

Further setup (database, uploads, AI models, Docker) will be appended here as each milestone adds it.
