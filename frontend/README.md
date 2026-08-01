# Frontend — Sports Deepfake Detection & Verification Platform

React + TypeScript SPA built with Vite and Tailwind CSS.

## Stack

- **React 19** + **TypeScript**
- **Vite** (dev server + build)
- **Tailwind CSS v4** (via `@tailwindcss/vite`, CSS-first config in `src/index.css`)
- **React Router** (Data Mode / `createBrowserRouter`)
- **Recharts** (Milestone 13 — the signal-breakdown chart on the results page)

## Setup

```bash
npm install
cp .env.example .env   # point VITE_API_BASE_URL at your backend
npm run dev            # http://localhost:5173
```

The backend must be running (see `../backend/README` instructions in the root docs) for the
Dashboard's connectivity check and all future data-driven pages to work.

## Structure

```
src/
├── components/
│   ├── analysis/     # AnalysisListItem — reusable row for lists of analyses
│   ├── layout/        # AppShell (header/nav/outlet)
│   ├── results/        # VerdictHeader, SignalChart, ReasonsList, NotesList — the results dashboard
│   ├── status/         # StatusPill, ProgressBar
│   └── upload/          # Dropzone
├── hooks/             # useBackendHealth, useFileUpload, useAnalysis (polling), useRecentAnalyses
├── lib/                # signalMeta.ts — display labels for detector signal names
├── pages/              # Dashboard, Upload, AnalysisDetail, History
├── services/           # api.ts — typed fetch client, one function per endpoint
├── store/               # reserved for state management once needed
└── types/               # shared TS types mirroring backend Pydantic schemas
```

### Results Dashboard (Milestone 13)

`/analysis/:id` polls `GET /analysis/{id}` every 2s while status is `pending`/`processing` (real
background-task processing takes actual time in a browser, unlike the backend's own synchronous
test client), then renders:
- `VerdictHeader` — headline, confidence %, risk level
- `ReasonsList` — the natural-language reasons from Milestone 10
- `SignalChart` — a Recharts horizontal bar chart of every detector's suspicion score, parsed
  from `detector_breakdown` JSON
- `NotesList` — which signals were unavailable and why

Uploading now redirects straight to this page instead of a static success card.

## Design system

Dark-first "broadcast review console" theme. Tokens (colors, fonts) are defined as CSS
variables in `src/index.css` under `@theme`, not hardcoded in components:

- `--color-authentic` / `--color-suspicious` / `--color-processing` — verdict states
- `--font-display` (Space Grotesk), `--font-body` (IBM Plex Sans), `--font-mono` (IBM Plex Mono)

## Scripts

- `npm run dev` — start dev server
- `npm run build` — type-check (`tsc -b`) and production build
- `npm run preview` — preview the production build locally
- `npm run lint` — oxlint
