# Sports Deepfake Detection & Verification Platform

A production-quality platform for detecting deepfakes and manipulated media in sports content — athlete interviews, match highlights, and broadcast footage — combining general-purpose deepfake forensics with sports-specific consistency checks (jerseys, logos, stadiums, crowds).

> Final Year Engineering Major Project — built incrementally, milestone by milestone.

## Status

🚧 **Milestone 1: Project Architecture & Folder Structure** — in progress.

See [`docs/architecture.md`](docs/architecture.md) for the system design and [`docs/milestones.md`](docs/milestones.md) for the build roadmap.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS, Recharts |
| Backend | FastAPI (Python) |
| AI / CV | PyTorch, OpenCV, MediaPipe, NumPy |
| Database | SQLite |
| Deployment | Docker (final milestone) |

## Project Structure

```
sports-deepfake-platform/
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/v1/           # Versioned REST endpoints
│   │   ├── core/             # Settings, logging, startup config
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── detection/    # Deepfake detection pipeline
│   │   │   ├── sports_intel/ # Sports-specific analysis
│   │   │   └── explainability/ # Explainable AI report generation
│   │   ├── db/                # DB session, migrations
│   │   └── utils/             # Shared helpers
│   └── tests/                 # Backend test suite
├── frontend/                  # React + TypeScript SPA
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── hooks/
│       ├── services/          # API client layer
│       ├── store/             # State management
│       └── types/
├── models/
│   ├── pretrained/            # Downloaded model weights (git-ignored)
│   └── configs/                # Model config/metadata (tracked)
├── database/                   # SQLite database file (git-ignored)
├── uploads/                    # User-uploaded images/videos (git-ignored)
├── reports/                    # Generated analysis reports (git-ignored)
├── docs/                       # Project documentation
├── scripts/                    # Setup / utility scripts
└── .github/workflows/          # CI pipelines (added later)
```

## Getting Started

Setup instructions will be added incrementally as each layer is built:

- Backend setup → added in Milestone 2
- Frontend setup → added in Milestone 3
- Full local run guide → `docs/installation.md` (grows each milestone)

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system architecture & data flow
- [`docs/api.md`](docs/api.md) — REST API reference (filled in as endpoints are built)
- [`docs/installation.md`](docs/installation.md) — local setup guide
- [`docs/deployment.md`](docs/deployment.md) — Docker deployment guide
- [`docs/milestones.md`](docs/milestones.md) — development roadmap & progress log

## License

Academic / educational project.
