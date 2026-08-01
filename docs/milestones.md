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
| 15 | Performance optimization | ⬜ Pending |
| 16 | UI polishing (dark mode, animations, responsiveness) | ⬜ Pending |
| 17 | Testing | ⬜ Pending |
| 18 | Docker deployment | ⬜ Pending |
| 19 | Documentation finalization | ⬜ Pending |
| 20 | Final project review | ⬜ Pending |

> Roadmap restructured after the Milestone 5 scope update to reflect the hybrid
> DL + forensics + fusion architecture. Milestones 1-6 are unaffected; 7 onward
> replace the original single "AI inference pipeline" step with a more honest
> breakdown of the actual hybrid pipeline.

Milestones may be split further into smaller sub-steps as needed (e.g. Milestone 8 may become 8a/8b/8c for each detector type).
