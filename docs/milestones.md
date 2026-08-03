# Development Roadmap

Progress log for the incremental build. Each milestone is completed in a single response and confirmed before moving to the next.

| # | Milestone | Status |
|---|---|---|
| 1 | Project architecture & folder structure | ✅ Done |
| 2 | Backend initialization (FastAPI skeleton, config, logging) | ✅ Done |
| 3 | Frontend initialization (Vite + React + TS + Tailwind) | ✅ Done |
| 4 | Database integration (SQLAlchemy models, SQLite) | ✅ Done |
| 5 | Upload system (image/video upload, drag-and-drop, progress) | ✅ Done |
| 6 | Media processing pipeline (frame extraction, preprocessing) | ✅ Done |
| 7 | Deep learning image detector (pretrained model, real/fake probabilities) | ✅ Done |
| 8 | Forensic analysis module (FFT, compression, landmarks, lighting) | ✅ Done |
| 9 | Fusion engine + background processing (pending→processing→completed) | ✅ Done |
| 10 | Explainable AI (human-readable reasons from raw signals) | ✅ Done |
| 11 | Video temporal extension (cross-frame consistency, optical flow, lip-sync) | ✅ Done (lip-sync deferred — needs audio pipeline, not yet in scope) |
| 12 | Sports-specific intelligence (jersey/logo/stadium/crowd/broadcast) | ✅ Done (athlete identity + match context deferred — need external reference data) |
| 13 | Results dashboard (frontend) | ✅ Done |
| 14 | History module | ✅ Done |
| 15 | Performance optimization | ✅ Done (SQLite WAL + gzip; lazy torch import cut the test suite 157s→25s) |
| 16 | UI polishing (dark mode, animations, responsiveness) | ⬜ Pending (dark theme exists; no light theme or toggle) |
| 17 | Testing | ✅ Done (162 backend tests, 91% coverage; **no frontend tests yet**) |
| 18 | Docker deployment | ⬜ Pending (R14) |
| 19 | Documentation finalization | ✅ Done (R15) |
| 20 | Final project review | ⬜ Pending |

> Roadmap restructured after the Milestone 5 scope update to reflect the hybrid
> DL + forensics + fusion architecture. Milestones 1-6 are unaffected; 7 onward
> replace the original single "AI inference pipeline" step with a more honest
> breakdown of the actual hybrid pipeline.

Milestones may be split further into smaller sub-steps as needed (e.g. Milestone 8 may become 8a/8b/8c for each detector type).


---

## Phase 2 — the engineering-review roadmap (R0–R15)

Milestones 1–20 above delivered a working *application*. A Principal-Engineer-level review of
that codebase then found the critical gap: **the platform produced confident verdicts with no
evidence it detected anything**, alongside a High-severity arbitrary-file-read vulnerability and
no version control at all. R0–R15 address that.

| # | Milestone | Status | Outcome |
|---|---|---|---|
| R0 | Version control & reproducibility | ✅ | `git init` — the project had **no** VCS; tooling config, dev/prod dependency split |
| R1 | Dependency resolution & first real run | ✅ | Original pins had **no Python 3.14 wheels**; first genuine end-to-end run, all 11 signals verified live |
| R2 | Dataset acquisition | ✅ | 1,967 images, licensing recorded, 51 distinct generators |
| R2.5 | **Confound audit** | ✅ | Caught container-only **ROC-AUC 1.0000** — the dataset was solvable from image width alone |
| R3 | Evaluation harness | ✅ | First real number: **0.4331, below chance** |
| R6 | Trained probe (transfer learning) | ✅ | **0.491 → 0.7534** |
| R4 | Learned calibration & fusion | ✅ | **0.7715** on its own inputs; **0.7402** end-to-end |
| R9 | Security hardening | ✅ | Arbitrary file read closed + 6 more findings; 19 regression tests |
| R11 | Alembic migrations | ✅ | Schema changes no longer require deleting the database |
| R12 | CI, lint, coverage gate | ✅ | 162 tests now actually run; lint 41 → 0; 91% coverage |
| R5 | Metadata provenance & C2PA | ✅ | Closes a stated PPT objective that had **zero** implementation |
| R7 | Visual XAI (heatmaps) | ⬜ | `architecture.md` claims "highlighted-frame evidence" that does not exist |
| R8 | Sports intelligence phase 2 | ⬜ | Pose plausibility is the best candidate |
| R10 | Job system & reliability | ⬜ | Records can stick in `processing`; frontend polls indefinitely |
| R13 | Reports & export | ⬜ | `reports/` exists; nothing writes user-facing reports |
| R14 | Docker deployment | ⬜ | Same as Milestone 18 |
| R15 | Documentation reconciliation | ✅ | This file, README, and the PPT divergence ADR |

### Three bugs worth remembering

Each *looked* like a working feature and silently did nothing. All three were found by running
the system and comparing outputs, not by tests:

1. **The learned fusion never ran in production.** The calibration named its first feature
   `"probe"` while the engine emitted `"trained_probe"`, so it silently fell back to the legacy
   weighted mean on every request.
2. **Provenance changed the verdict by exactly nothing.** It ran, appeared in the breakdown with
   a weight, produced a reason line — and the fitted combiner ignored it. Caught by scoring two
   files with identical pixels and different metadata.
3. **A genuine Nikon D810 photo was flagged AI-generated**, because its XMP contained
   `aux:imagenumber="177034"` and `imagen` is a generator marker.
