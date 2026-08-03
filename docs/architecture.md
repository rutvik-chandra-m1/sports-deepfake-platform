# Architecture

## Overview

The platform is a client-server application with a clear separation between:

1. **Frontend (React + TypeScript + Vite)** — upload UI, results dashboard, history browser.
2. **Backend (FastAPI)** — REST API, orchestrates media processing and AI inference.
3. **AI/CV Pipeline** — a set of composable detectors (face artifact, temporal, compression, landmark, frequency-domain, lip-sync) plus sports-specific analyzers, combined into a single verdict with an explainability layer.
4. **Persistence (SQLite)** — stores analysis history, results, and metadata.
5. **File Storage (local disk under `uploads/` and `reports/`)** — holds raw media and generated reports; abstracted behind a storage service so it can later be swapped for cloud storage without touching business logic.

## High-Level Data Flow

```
┌─────────────┐     upload      ┌──────────────┐     enqueue      ┌──────────────────┐
│   Frontend   │ ──────────────▶ │  FastAPI API  │ ────────────────▶ │  Analysis Pipeline │
│ (React/Vite) │                 │  (backend)    │                   │  (services layer)  │
└─────────────┘ ◀────────────── └──────────────┘ ◀──────────────── └──────────────────┘
     ▲   poll status / fetch report        │                                │
     │                                     ▼                                ▼
     │                              ┌─────────────┐                 ┌──────────────┐
     └─────────────────────────────│   SQLite    │◀────────────────│  models/ (AI) │
                                    │  (history)  │                 │  weights      │
                                    └─────────────┘                 └──────────────┘
```

## Backend Layering (`backend/app/`)

- **api/v1/** — thin HTTP-facing routers. No business logic; validates input via `schemas/` and delegates to `services/`.
- **core/** — application settings (env-driven config), logging setup, app startup/shutdown hooks.
- **models/** — SQLAlchemy ORM models (database tables).
- **schemas/** — Pydantic models for request/response validation, decoupled from ORM models.
- **services/detection/** — the core deepfake forensics pipeline. `image_detector.py` (Milestone 7) wraps a real pretrained ViT deepfake classifier, extended in Milestone 11 (`predict_video`) to reason across multiple sampled video frames rather than just the first. `frequency_analysis.py`, `compression_analysis.py`, `lighting_analysis.py`, `landmark_analysis.py` (Milestone 8), and `optical_flow_analysis.py` (Milestone 11) are classical CV forensic signals with no trained weights; `forensic_analysis.py` orchestrates all of them, degrading individual failures to a non-applicable signal rather than aborting the batch. See `docs/models.md` for exact techniques, cited metrics (where applicable), licenses, and limitations.
- **services/media_processing/** — shared, model-agnostic input stage used by every detector: frame extraction (single frame for images, evenly-spaced sampling across the full duration for video), metadata (dimensions/fps/duration), and generic preprocessing (resize + RGB + [0,1] scaling). Detector-specific normalization happens in the detectors themselves, not here.
- **services/sports_intel/** — sports-specific verification (Milestone 12): jersey/clothing color consistency, background scene consistency, broadcast overlay tampering, and crowd-texture duplication. All classical CV, no sports-specific pretrained models (none exist off-the-shelf). Checks internal self-consistency of the uploaded media, not identification against real teams/stadiums/broadcasters. See `docs/models.md` for scope and what was deliberately left out (athlete identity verification, match context verification — both need external reference data this project doesn't have).
- **services/explainability/** — turns raw detector outputs into human-readable explanations.
  **NOT YET IMPLEMENTED: visual evidence.** This document previously claimed "highlighted-frame evidence"; no heatmap, bounding box or annotated frame is produced. The layer is text-only. Tracked as R7.
- **services/provenance/** — metadata provenance (R5): EXIF/IPTC generator markers and C2PA
  Content Credentials. Reads the FILE, not decoded frames, because metadata lives in the
  container and is destroyed on decode. Absence of metadata is never treated as evidence of
  manipulation.
- **services/fusion_calibration.py** — applies fusion weights fitted on held-out data (R4), with
  provenance layered on as a log-odds adjustment. Falls back to the legacy weighted mean when the
  calibration is absent or a required signal is missing.
- **core/security.py** — path containment, upload content validation, API-key auth, rate
  limiting, production-readiness startup checks (R9).
- **db/** — database session management. Schema evolution is owned by **Alembic** (`backend/alembic/`),
  not `create_all` (R11).
- **utils/** — shared helpers (file validation, video frame extraction, etc.).

## Design Principles

- **Single Responsibility**: each detector is an independent, swappable module implementing a common interface (`BaseDetector`), so new detection techniques can be added without touching existing ones.
- **Composability**: the pipeline aggregates multiple detector outputs into a single weighted verdict, rather than relying on one monolithic model.
- **Explainability-first**: every detector emits both a score and a human-readable reason, not just a number.
- **Config over code**: model paths, thresholds, and enabled detectors are controlled via configuration (`models/configs/`, `.env`), not hardcoded.
- **Local-first storage**: SQLite + local disk keep the project runnable with zero external infra for a final-year project, while the storage/service layers stay abstracted enough to swap in Postgres/S3 later.

## Planned API Surface (implemented incrementally)

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/media/upload` | Upload image/video for analysis |
| `POST /api/v1/analysis/{analysis_id}/run` | Re-run analysis on an existing record |
| _(not built)_ `GET /api/v1/analysis/{id}/status` | Lightweight polling; the UI currently re-fetches the whole record every 2s |
| `GET /api/v1/analysis/{analysis_id}` | Get full report/result |
| `GET /api/v1/analysis` | List past analyses (there is no `/history` route — the UI's History page calls this) |
| `DELETE /api/v1/analysis/{analysis_id}` | Delete a record |

Full request/response contracts will be documented in `docs/api.md` as each endpoint is implemented.
