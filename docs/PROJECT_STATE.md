# PROJECT STATE — living handoff document

**Read this first in any new session.** It carries the context that would otherwise be lost:
what the project is, what actually works (with measured numbers), the decisions already made and
why, the traps already hit, and what remains.

Last updated: 2026-08-04, after R14. **All 16 roadmap items are resolved**: R0–R7 and R9–R15 are
implemented and verified; **R8 is deliberately deferred** with a written gate
([ADR 0002](adr/0002-defer-pose-plausibility-signal.md)) because the sports test split (n=10)
cannot measure it.

State: 183 tests passing, ruff clean, ~91% coverage, both Docker images built and the full stack
exercised end to end.

---

## 1. What this project is

**Title (academic):** *Detection of AI-Generated Sportsman Images Using Deep Learning* — KSIT
final-year major project, Phase-I (BCS685), Group 05.

**Repo:** a FastAPI + React platform that ingests an image or video and returns an
authentic/suspicious verdict with per-signal explanation.

### ⚠️ The PPT and the code describe different projects

| PPT commits to | Repo actually does | Status |
|---|---|---|
| **CNN** (ResNet/Xception lineage) | **Vision Transformer** (ViT-B/16) | Defended in **ADR 0001** |
| **TensorFlow** | **PyTorch** | Defended in **ADR 0001** |
| **C2PA metadata provenance** | Implemented in R5 | ✅ **Resolved** |
| Data collection → feature extraction → classification | Real dataset + trained probe | ✅ **Resolved** |
| Images only | Images **and** video | Video is unevaluated and reported as such |

**Read `docs/adr/0001-vit-pytorch-instead-of-cnn-tensorflow.md` before the viva.** It carries the
argument, the anticipated questions, and two acceptable resolutions to agree with the guide:
revise the PPT, or add a CNN baseline (the stronger submission — the dataset, splits and harness
already exist, so only a training script is new).

---

## 2. Current state — what actually works

### Headline: the pipeline detects. This was NOT true before R4/R6.

| Stage | test ROC-AUC | Note |
|---|---:|---|
| Hand-weighted fusion (R3 measurement) | **0.4331** | **below chance** — the original system did not work |
| Stock third-party face-tuned ViT head | 0.4910 | chance; it is trained on faces, data is not faces |
| Our trained probe, 318 training images | 0.6110 | |
| **Our trained probe, 1194 training images** | **0.7534** | CI 0.693–0.808 |
| **Learned fusion (probe + classical)** | **0.7715** | CI 0.716–0.824, val 0.7512 — agrees, generalises |

**Bars it must beat** (never compare to 0.50 alone):
- majority-class accuracy **0.5382**
- content-statistics baseline **0.5458** (ALL) / 0.5323 (general) / 0.8350 (sports)

**The single change that mattered was more training data.** Nothing architectural differs
between the 0.611 and 0.753 rows — only 318 → 1194 images. More data remains the highest-value
improvement available.

### Qualitative failures from R3 that are now fixed

| R3 (broken) | Now |
|---|---|
| Flagged 74% suspicious vs 48% actual | 53.5% vs 53.8% actual |
| Real images scored *more* suspicious than fakes (0.551 vs 0.523) | Correct ordering (0.449 vs 0.582) |
| Scores crushed into [0.158, 0.737] around threshold | Spread [0.147, 0.845] |

### What is still weak — do not overstate

- **Sports domain is effectively unmeasured**: n=10 in test. It reports 1.0 AUC. **That number is
  noise. Never quote it.**
- **Classical forensic signals are individually at or below chance** and their direction is *not
  stable across splits*. The trained probe does essentially all the work; fusion adds ~0.02.
- **5 of 11 signals never fire on a still image** (landmark, optical flow, temporal, jersey,
  scene are video-only). For the PPT's image scope this is a 6-signal system.
- Some probe overfitting remains (train 0.837 vs test 0.753).

---

## 3. Environment — non-obvious and easy to get wrong

- **OS/Python:** Windows 10, **Python 3.14**. Several original pins had no 3.14 wheels; `numpy`,
  `pydantic`, `pydantic-settings`, `SQLAlchemy` were bumped for this (see `requirements.txt`).
- **No GPU.** Intel i7-8665U, 4 cores, 15.8 GB RAM (~6–7 GB typically free). This is why R6 uses
  a **linear probe on frozen features**, not fine-tuning.
- **Network is slow and highly variable** (~100 kB/s to ~1 MB/s). Long downloads belong in
  background tasks.
- **TWO separate virtualenvs — this trips people up constantly:**

| venv | Purpose | Has |
|---|---|---|
| `backend/.venv` | production API + anything importing `app.*` | torch, transformers, opencv, fastapi |
| `ml/.venv` | data prep, training, evaluation metrics | scikit-learn, datasets, diffusers, matplotlib |

`ml/eval/run_inference.py` and `ml/train/extract_embeddings.py` need the **backend** venv (they
import the real pipeline). `evaluate.py`, `train_probe.py`, `train_fusion.py` need the **ml**
venv. Getting this wrong produces confusing ImportErrors.

- **Repo root:** `D:\MAJOR PROJECT\sports-deepfake-platform-current\sports-deepfake-platform`
  (git repo, branch `master`). Note the doubled directory name.
- **Background task output is fully buffered** — a running job often shows an empty log until it
  finishes. Check the process list or output files for progress, not the log.

---

## 4. Traps already hit — do not repeat these

1. **Dataset confounds are invisible unless tested for.** The first dataset build was perfectly
   separable using **zero pixels** — container-only ROC-AUC **1.0000** (`width > 640` solved it).
   Going straight to evaluation would have reported ~99% accuracy measuring *file dimensions*.
   **Always run `audit_dataset.py` before trusting any metric.**
2. **`Parveshiiii/AI-vs-Real` is unusable.** All reals exactly 178×218, all fakes exactly
   1024×1024 — class and source perfectly correlated, unfixable by normalization (equalising
   requires upscaling one class). Withdrawn. Current backbone is **`ComplexDataLab/OpenFake`**.
3. **Verify labels before concluding a model is broken.** When 5/6 signals looked inverted, a
   label flip was the obvious suspect — I checked images visually. Labels were correct.
4. **Never select a model on the test split.** PCA scored better on val but worse on test; val
   is what decides. Choosing on test is fishing.
5. **Never fit fusion on train.** The probe's train scores are in-sample; fusion is fitted on
   **val** only.
6. **Small val sets fit noise.** Fusion on 113 val rows *anti-generalised* (test 0.447). On 321
   rows it worked (0.7715). Signals whose direction flips across splits are noise, not evidence.
7. **`.env` paths were CWD-relative** — running a script from a different directory silently
   created a second `models/pretrained` tree and re-downloaded ~330 MB. Fixed by anchoring to
   `backend/` in `config.py`; keep it that way.
8. **Large HF fetches hit `MemoryError`** (camera-resolution images + shuffle buffer). Fixed by
   writing the manifest **incrementally** and bounding image size via `--max-side`. An earlier
   crash orphaned ~500 downloaded images by losing the manifest.
9. **`diffusers`' `AutoPipelineForText2Image` is broken under `transformers` 5.x** (eagerly
   imports a HunyuanDiT pipeline referencing a removed class). Use `StableDiffusionPipeline`
   directly.
10. **Mocked pipeline tests must pin the probe.** It carries the largest fusion weight; unmocked
    it runs the real model and legitimately outvotes the mocked signals.
11. **Artifacts crossing a process boundary degrade SILENTLY.** Three separate features looked
    like they worked and did nothing. None were caught by tests; all were caught by running the
    system and comparing outputs:
    - the calibration named its first feature `"probe"` while the engine emitted
      `"trained_probe"` → the learned fusion **never ran in production**
    - provenance ran, appeared in the breakdown with a weight, produced a reason line → and
      changed the fused score by **exactly nothing**, because the fitted combiner consumes only
      its own features
    - **Always verify a new signal changes the output**, e.g. score two files with identical
      pixels and different metadata. `tests/test_fusion_calibration.py` now locks both contracts.
12. **Substring matching over metadata produces false accusations.** A genuine Nikon D810 photo
    was flagged AI-generated because its XMP contained `aux:imagenumber="177034"` and `imagen` is
    a generator marker. Match whole tokens, and only in software-identifying fields.

---

## 5. Where things live

```
backend/app/services/
  analysis_pipeline.py      analyze_frames() = PURE detection (no DB) + BackgroundTask wrapper
  fusion_engine.py          combines signals; prefers learned calibration, falls back to legacy
  fusion_calibration.py     applies the fitted combiner (numpy only, no sklearn at serve time)
  detection/
    probe_detector.py       OUR trained classifier -- the primary detector
    image_detector.py       third-party face-tuned ViT (kept, low weight, ~chance on this data)
    *_analysis.py           classical heuristics (at/below chance individually)
  sports_intel/             sports checks (self-consistency only, not identity verification)
  explainability/reasoning.py   templated natural-language verdict

ml/                         SEPARATE venv -- data, training, evaluation
  data/  fetch_hf_backbone.py  fetch_wikimedia_sports.py  generate_synthetic.py
         build_manifest.py  audit_dataset.py  normalize_dataset.py  probe_hf_dataset.py
  train/ extract_embeddings.py  train_probe.py  train_fusion.py
  eval/  run_inference.py (backend venv!)  evaluate.py (ml venv)

models/configs/probe_head.json          trained head -- plain JSON, tracked in git
models/configs/fusion_calibration.json  learned fusion weights -- tracked
datasets/manifest_normalized.csv        USE THIS for all training/eval
reports/evaluation/                     predictions.csv, metrics JSON, plots
```

**Key docs:** `evaluation.md` (results), `dataset.md` (composition + confound audit),
`models.md` (every model + limitations), `architecture.md`, `milestones.md`.

---

## 6. Commands that work

```bash
# Tests (183 passing, ~91% coverage)
cd backend && .venv/Scripts/python.exe -m pytest -q

# Whole stack in Docker -- needs API_KEY + SECRET_KEY in ./.env
docker compose up --build     # UI on http://localhost:8080

# Run the app (local dev)
cd backend && .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
cd frontend && npm run dev            # http://localhost:5173

# Full data -> model -> evaluation loop
cd ml/data
../.venv/Scripts/python.exe fetch_hf_backbone.py --per-class 1000 --shuffle-buffer 600
../.venv/Scripts/python.exe build_manifest.py
../.venv/Scripts/python.exe normalize_dataset.py
../.venv/Scripts/python.exe audit_dataset.py --manifest manifest_normalized.csv   # MUST be ~0.50
cd ../train
../../backend/.venv/Scripts/python.exe extract_embeddings.py      # BACKEND venv, ~0.42s/image
../.venv/Scripts/python.exe train_probe.py
../.venv/Scripts/python.exe train_fusion.py
cd ../eval
../../backend/.venv/Scripts/python.exe run_inference.py           # BACKEND venv, ~0.8s/image
../.venv/Scripts/python.exe evaluate.py --split test
```

---

## 7. Roadmap — done and remaining

### Done
| | Milestone | Outcome |
|---|---|---|
| R0 | Git + environment reproducibility | repo initialised (there was **no** version control) |
| R1 | Dependency resolution + first real run | Python 3.14 wheel conflicts fixed; 11 signals verified live |
| R2 | Dataset acquisition | 3 sources, licensing recorded |
| R2.5 | **Confound audit** | caught AUC 1.0000 leak; built audit/normalize/probe tooling |
| R3 | Evaluation harness | first real number: **0.4331, below chance** |
| R6 | Trained probe | 0.491 → **0.7534** |
| R4 | Learned fusion | **0.7715** on its own inputs; **0.7402** end-to-end |
| R9 | Security hardening | arbitrary file read + 6 more findings closed; 19 regression tests |
| R11 | Alembic migrations | schema changes no longer destroy the database |
| R12 | CI, lint, coverage | 162 tests now actually run; lint 41→0; 91% coverage |
| R5 | Provenance & C2PA | closes a stated PPT objective that had zero implementation |
| R15 | Documentation | README/milestones/architecture/installation/api reconciled; ADR 0001 written |
| R7 | Visual XAI | attention rollout + per-signal overlays; fixed a cubic-overshoot bug that rendered the hottest regions **dark** |
| R10 | Job system | bounded pool, stale-record recovery, frontend geometric backoff with a 5-min ceiling |
| R13 | Reports & export | PDF + JSON; limitations render on **page 1, before the verdict** |
| R14 | Deployment | backend + frontend images, compose, nginx; both images built and exercised |

### Remaining

**Only R8 remains, and it is deliberately deferred — see
[ADR 0002](adr/0002-defer-pose-plausibility-signal.md).**

**R8 — Sports intelligence phase 2 (pose plausibility).** Not implemented. The sports test split
is **n=10** (3 real / 7 fake) with a bootstrap 95% CI of 0.000–0.556, so any change it produced
would be indistinguishable from noise. Worse, fusion renormalises across applicable signals, so
adding an unvalidated signal would take weight away from `trained_probe` — the only component with
a measured above-chance result. The gate to unblock it is written down in the ADR: ≥200 sports test
images, passing the confound audit at both tiers, with fakes from ≥2 generators.

**Highest-value single action for detection quality: more training data.** Everything else is
engineering, not research risk. That conclusion has survived every milestone since R3.

---

## 8. Working agreements established with the user

- Implement milestone by milestone; **pause for approval after each**.
- Don't regenerate existing files; edit them.
- Report failures honestly with real output — never claim something works unverified.
- Prefer measuring over asserting: every number in the docs came from a real run.
