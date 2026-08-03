# Sports Deepfake Detection & Verification Platform

Detects AI-generated and manipulated media in sports content — athlete photos, match imagery and
broadcast footage — by combining a trained classifier, classical image forensics, metadata
provenance verification, and sports-specific consistency checks.

> Final Year Engineering Major Project.

## Status — it works, and here is the number

**Held-out test set (n=275): ROC-AUC 0.7402, accuracy 71.3%.**

| Metric | Value | Bar it must beat |
|---|---:|---|
| ROC-AUC | **0.7402** (95% CI 0.680–0.796) | 0.5458 — content-statistics baseline |
| Accuracy | **71.3%** | 53.8% — majority class |
| Precision / Recall (fake) | 73.5% / 73.0% | — |

That number is deliberately the first thing in this README, because the honest version of this
project's story is that **it did not work at first**. The initial hand-weighted pipeline measured
**ROC-AUC 0.4331 — below chance**. See [`docs/evaluation.md`](docs/evaluation.md) for the full
measurement, and [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for how it got from there
to here.

**This is a research prototype, not a deployable moderation tool.** A ~29% error rate is far too
high to accuse anyone of faking an image. State-of-the-art detection is 0.95+.

### What is measured, and what is not

- ✅ **General imagery**: ROC-AUC 0.7306 (n=265), the number that carries weight.
- ❌ **Sports-specific**: the test split has only **n=10**. It reports 1.0 AUC. **That figure is
  noise and must not be quoted.**
- ❌ **Video**: 5 of 11 signals are video-only and are unevaluated — the dataset is images.

## How it works

```
upload → path containment + magic-byte validation → frame extraction
       → trained probe (ours)  ─┐
       → classical forensics   ─┼→ learned fusion → verdict + explanation
       → sports intelligence   ─┤   (calibrated on held-out data)
       → metadata provenance   ─┘
```

| Layer | What it contributes |
|---|---|
| **Trained probe** (ours) | The primary detector. ViT-B/16 trunk + a logistic head trained on this project's dataset. **Test ROC-AUC 0.7534** |
| Classical forensics | FFT spectra, Error Level Analysis, lighting, landmarks, optical flow. Individually **at or below chance** on this data — kept as visible corroboration, not authority |
| Sports intelligence | Jersey/scene consistency, broadcast overlay, crowd-texture duplication. Self-consistency only, no reference database |
| **Metadata provenance** | EXIF/IPTC generator markers and C2PA Content Credentials. **Absence is never treated as evidence of manipulation** |
| Learned fusion | Weights fitted on a held-out split, so inverted signals get negative coefficients instead of being summed the wrong way |

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind v4, Recharts |
| Backend | FastAPI, SQLAlchemy 2, Alembic, SQLite (WAL) |
| AI / CV | PyTorch, Transformers (ViT), OpenCV, MediaPipe |
| ML tooling | scikit-learn, HF Datasets, Diffusers *(separate venv — see `ml/`)* |
| Provenance | Pillow (EXIF/XMP), c2pa |
| Quality | pytest (**162 tests, 91% coverage**), ruff, GitHub Actions |

## Quick start

```bash
# Backend
cd backend
python -m venv .venv && .venv\Scripts\Activate.ps1   # or source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head            # schema via migrations, never by deleting the DB
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

App at http://localhost:5173, API docs at http://localhost:8000/docs.
Full guide: [`docs/installation.md`](docs/installation.md).

> **Two virtualenvs by design.** `backend/.venv` runs the API; `ml/.venv` runs data prep,
> training and evaluation. Mixing them causes confusing ImportErrors — see [`ml/README.md`](ml/README.md).

## Reproducing the results

```bash
cd ml/data
python fetch_hf_backbone.py --per-class 1000    # dataset
python build_manifest.py && python normalize_dataset.py
python audit_dataset.py --manifest manifest_normalized.csv   # MUST be ~0.50, see below

cd ../train
../../backend/.venv/Scripts/python.exe extract_embeddings.py
python train_probe.py && python train_fusion.py

cd ../eval
../../backend/.venv/Scripts/python.exe run_inference.py
python evaluate.py --split test
```

**Always run the audit before trusting a metric.** The first dataset build was perfectly
separable using **zero pixels of content** — image width alone gave ROC-AUC **1.0000**. Going
straight to evaluation would have reported ~99% accuracy that measured file dimensions.
See [`docs/dataset.md`](docs/dataset.md).

## Documentation

| Doc | What it covers |
|---|---|
| [`PROJECT_STATE.md`](docs/PROJECT_STATE.md) | **Start here** — current state, decisions, traps, roadmap |
| [`evaluation.md`](docs/evaluation.md) | Measured results, per-signal ablation, baselines |
| [`dataset.md`](docs/dataset.md) | Composition, licensing, and the confound audit |
| [`models.md`](docs/models.md) | Every model, with limitations |
| [`security.md`](docs/security.md) | 13 findings: 7 fixed, 6 open, plus threat model |
| [`architecture.md`](docs/architecture.md) | System design and data flow |
| [`api.md`](docs/api.md) · [`installation.md`](docs/installation.md) · [`deployment.md`](docs/deployment.md) | Reference guides |
| [`milestones.md`](docs/milestones.md) | Build log |

## Known gaps

Recorded rather than hidden — see [`PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the full list.

- **Sports-domain performance is effectively unmeasured** (n=10).
- **No per-user authentication** — a shared API key authenticates the client, not a user.
- Model supply chain is unpinned (no revision SHA or checksum on downloaded weights).
- Uploaded files are never deleted; no retention policy.
- No Docker deployment yet; no visual explanation (heatmaps) yet.

## License

Academic / educational project.
