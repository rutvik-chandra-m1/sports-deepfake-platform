# AI Models

## Trained linear probe — this project's own classifier (R6)

**The primary detector.** Everything else on this page is a third-party model or a
hand-written heuristic; this is the one component trained on this project's own labelled data,
and the only one measured above chance on a held-out split.

**Why it exists.** R3 measured the third-party face-tuned ViT head (below) at ROC-AUC **0.491 —
chance** — on this dataset, exactly the out-of-distribution failure this document had warned
about: it is trained on faces, and the data is general and sports imagery. Rather than discard
the model, the probe keeps its **trunk** (`google/vit-base-patch16-224-in21k`, a general-purpose
backbone) and replaces only the mismatched classification head.

| Property | Value |
|---|---|
| Architecture | Frozen ViT-B/16 trunk → PCA(128) → L2-regularised logistic head |
| Trained on | 1,194 images (`datasets/manifest_normalized.csv`, train split) |
| Hyperparameters | `n_components` and `C` chosen by 5-fold CV **inside train** |
| Artifact | `models/configs/probe_head.json` (~50KB of plain JSON) |
| Trained by | `ml/train/train_probe.py` |
| Served by | `backend/app/services/detection/probe_detector.py` |

### Measured performance (held-out test, n=275)

| Split | n | ROC-AUC | 95% CI |
|---|---:|---:|---|
| train | 1194 | 0.8366 | 0.814–0.859 |
| val | 321 | 0.7391 | 0.690–0.794 |
| **test** | **275** | **0.7534** | **0.693–0.808** |

Test accuracy **0.7200** at threshold 0.5 (0.6982 at the val-selected 0.5642). The CI sits
clearly above both chance and the 0.5458 content-statistics baseline from the dataset audit, so
this is a real effect, not noise. Per-domain on test: general **0.7437** (n=265); the sports
subset reports 1.0 but at **n=10 that number is meaningless** and must not be quoted.

### Why training-set size mattered more than anything else

| Training images | test ROC-AUC |
|---:|---:|
| 0 (stock face head) | 0.491 |
| 318 | 0.611 |
| **1194** | **0.7534** |

The first attempt overfit badly (train 0.974 vs val 0.612 on 318 images). PCA plus more data
closed that gap to 0.837 vs 0.753 — still some overfitting, so **more data remains the single
highest-value improvement**, ahead of any architectural change.

### Design decisions worth knowing

- **Linear probing, not full fine-tuning.** No GPU is available (Intel i7-8665U). A frozen
  backbone needs one forward pass per image (0.42s measured) and the head then trains in
  milliseconds; backpropagating through ViT-B/16 on CPU is not practical. Linear probing is the
  standard way to measure what a frozen representation already encodes.
- **Exported as plain JSON, not a pickle.** The backend applies the head with a single numpy dot
  product. This keeps scikit-learn out of the serving dependency tree entirely, and avoids
  unpickling a file from disk — pickle deserialisation is arbitrary code execution.
- **PCA and the head are algebraically collapsed** into one linear map in the original 768-d
  space before export, and the collapse is **verified against the sklearn pipeline** (max logit
  drift 7.1e-07) so training and serving cannot silently diverge. Independently re-verified after
  export: the JSON reproduces val 0.7391 / test 0.7534 exactly.
- **Degrades gracefully.** If `probe_head.json` is absent (e.g. a fresh clone before training),
  the detector returns a non-applicable signal and the pipeline continues without it, exactly
  like every other detector.

### Limitations

- Trained on 1,194 pilot-scale images. Generalisation beyond this dataset's distribution is
  unmeasured.
- The backbone is still face-pretrained-adjacent; a backbone pretrained on a broader corpus
  might probe better.
- **Sports-domain performance is effectively unmeasured** (n=10 in test).
- Threshold 0.5642 was selected on val by Youden's J. A deployment with asymmetric costs (a false
  accusation against a real athlete is worse than a missed fake) should re-select it against an
  explicit cost ratio.


This document tracks every pretrained model the platform integrates: what it is, why it was
chosen, what it actually reports (cited from the model's own card — we do not invent or
independently verify accuracy numbers unless explicitly stated), its license, and its known
limitations.

## Image-level deepfake detector (Milestone 7)

**Model:** [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model)
**Architecture:** Vision Transformer (`google/vit-base-patch16-224-in21k` base), fine-tuned for
binary classification.
**License:** Apache-2.0
**Input:** RGB images, resized to 224×224 by the model's own processor.
**Output:** `Realism` vs `Deepfake` (normalized in our code to `real_probability` /
`fake_probability`, label-order-independent).

### Reported metrics (from the model card, not independently reproduced by us)

```
              precision    recall  f1-score   support
     Realism     0.9683    0.8708    0.9170     28001
    Deepfake     0.8826    0.9715    0.9249     28000
    accuracy                         0.9212     56001
```

The model card describes this as evaluated on the author's own curated real/deepfake face
dataset. We have not re-run this evaluation ourselves — treat it as the authors' reported
number, not a platform-verified guarantee.

### Why this model

- Real, publicly documented fine-tuning (not just an ImageNet backbone with an untrained head)
- Transfer learning from a well-known base (ViT-B/16, `google/vit-base-patch16-224-in21k`),
  matching the "use pretrained models, transfer learning where appropriate" requirement
- Permissive license (Apache-2.0) suitable for an academic project
- Straightforward integration via `transformers.AutoImageProcessor` /
  `AutoModelForImageClassification` — no custom architecture code to maintain

### Known limitations (from the model card + our own read)

- Trained on **face imagery**. Sports content that isn't a face close-up (crowd shots,
  broadcast graphics, wide match footage) is out of this model's training distribution — its
  output there should be treated as low-confidence until Milestone 9's fusion engine weighs it
  against the forensic signals (Milestone 8) and, later, sports-specific checks (Milestone 12).
- May not generalize to deepfake generation methods not represented in its training data.
- Performance may degrade on low-resolution or heavily compressed footage — exactly the kind
  of media a fusion engine should cross-check against compression-artifact analysis (M8).
- Image-only; does not itself capture temporal/video-level artifacts (Milestone 11 adds that).

### Runtime requirement — read this before running Milestone 7 code

Weights (~330MB) download from Hugging Face Hub **the first time `predict()` runs**, cached
under `models/pretrained/` (`MODELS_DIR` in `.env`). This needs outbound internet access.

**R1 (2026-08-01) confirmed this for real** on a Windows 11 dev machine (Python 3.14, real
internet access — not the sandbox this project was originally built in, see history below).
`model.safetensors` (343,223,968 bytes, matching the HF repo's listed 343MB exactly) downloaded
and cached under `models/pretrained/models--prithivMLmods--Deep-Fake-Detector-v2-Model/`. A real
synthetic image run through the live API produced a genuine `deep_learning` score (0.577) inside
`detector_breakdown.signals` — not an `unavailable` entry. Total pipeline latency: ~10s
(model already warm from the test suite's own real-download test, see below).

**Windows-specific note:** `huggingface_hub`'s cache uses symlinks by default; without Developer
Mode or an elevated shell, Windows can't create them, so it falls back to copying full files into
`snapshots/` instead of blob+symlink (works fine, just uses more disk on repeat downloads of the
same model at different revisions). `huggingface_hub` prints a one-time warning about this —
harmless, silence it by setting `HF_HUB_DISABLE_SYMLINKS_WARNING=1` if it's noisy, or enable
Developer Mode (Settings → Privacy & security → For developers) to get real symlinks.

**Original sandbox history, kept for context:** this project was originally built in a sandboxed
dev environment allowlisted to package registries only (PyPI, npm, GitHub), not `huggingface.co`
— `predict()` would raise:

```
OSError: Can't load image processor for 'prithivMLmods/Deep-Fake-Detector-v2-Model'.
If you were trying to load it from 'https://huggingface.co/models', ...
```

That was an environment restriction, not a bug — `predict()` catches it and raises a clear
`ModelLoadError` rather than crashing (see `tests/test_image_detector.py`, which verifies this
error handling with a locally-constructed mock model, no network required). That fallback path
is still real and still tested; it's just no longer the only evidence this pipeline works.

To verify it yourself:

```bash
cd backend && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -c "
import cv2
from app.services.detection.image_detector import predict

image_bgr = cv2.imread('/path/to/any/photo.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = predict(image_rgb)
print(result)
"
```

First run downloads the model (~330MB, one-time); subsequent runs are fast and fully offline.

## Classical forensic detectors (Milestone 8)

None of these use pretrained/trained weights in the machine-learning sense — they're
well-established signal-processing/CV techniques. Their `suspicion_score` is a documented
heuristic, not a calibrated probability; see `app/services/detection/types.py::ForensicSignal`.

| Detector | Technique | External asset? |
|---|---|---|
| `frequency_analysis.py` | FFT radial power-spectrum "bumpiness" (periodic/GAN-upsampling artifacts show up as spectral peaks that smooth natural photos don't have) | None — pure NumPy/OpenCV |
| `compression_analysis.py` | Error Level Analysis (ELA): block-wise coefficient of variation after JPEG re-encoding | None — pure OpenCV |
| `lighting_analysis.py` | Face-region vs background brightness/color-balance comparison (falls back to whole-image quadrant check if no face found) | Haar cascade XML (~930KB), downloaded once from `raw.githubusercontent.com/opencv/opencv` and cached under `models/pretrained/haarcascades/` |
| `landmark_analysis.py` | MediaPipe Face Landmarker — frame-to-frame landmark displacement ("jitter"), video only (needs 2+ frames) | Face Landmarker `.task` bundle (~a few MB), downloaded once from `storage.googleapis.com/mediapipe-models` and cached under `models/pretrained/mediapipe/` |

**Environment note:** both external assets above download from domains outside the original
build sandbox's allowlist (`raw.githubusercontent.com` actually *was* reachable there, so the
Haar cascade was downloaded and tested for real from day one; `storage.googleapis.com` was not,
so `landmark_analysis.py`'s network path was originally tested only via dependency injection +
one real confirmed failure). **R1 (2026-08-01) confirmed both work for real** on an unrestricted
connection: `haarcascade_frontalface_default.xml` (912KB) and `face_landmarker.task` (3.6MB)
both downloaded and cached under `models/pretrained/`. More importantly, both ran real
*inference*, not just download — a real synthetic video pushed through the live API produced a
genuine `landmark_instability` score (0.517) and a genuine `jersey_color_consistency` score
(0.054) in `detector_breakdown.signals`, meaning MediaPipe's landmarker and the Haar cascade each
detected a face and tracked it across frames for real, not just confirmed network reachability.

## Fusion engine (Milestone 9)

Combines the Milestone 7 DL detector output and Milestone 8's four forensic signals into one
verdict. Implementation: `app/services/fusion_engine.py`.

**Weighting scheme** (a documented design choice, not fit/calibrated against a labeled
validation set): the DL detector gets nominal weight 0.5 (it's the only signal here with a
published, cited evaluation — see the ViT model above); the remaining 0.5 is split evenly
across the four forensic signal slots (0.125 each). Whichever signals are actually applicable
have their nominal weights renormalized to sum to 1 — so a missing/unavailable signal (e.g. no
internet for the DL model or the landmark model) doesn't leave weight on the floor; the
remaining signals simply account for 100% of the decision. All thresholds (verdict cutoff,
risk-level bands) are configurable via `.env` (`FUSION_*` settings), not hardcoded.

**Explanation text** generated here is intentionally minimal (which signals contributed, at
what weight, which were unavailable) — Milestone 10 replaces this with a proper natural-language
explainability layer, reading the same `detector_breakdown` JSON this engine writes to the
`Analysis.detector_breakdown` column.

**Background processing:** `app/services/analysis_pipeline.py::run_analysis_pipeline` runs the
whole thing (load media → DL detector → forensic analysis → fuse → persist) as a FastAPI
`BackgroundTask`, triggered by `POST /media/upload` and by the new `POST /analysis/{id}/run`
(for reprocessing). A record's `status` moves `pending → processing → completed`/`failed`. Any
single detector failing (most commonly the DL model or landmark model needing network access)
degrades that one signal to "unavailable" rather than failing the whole analysis — confirmed by
actually running real uploads (image and video) through the full API in this sandbox, where the
DL and landmark signals are unavailable (no Hugging Face Hub / Google storage access here) but
the three network-free forensic signals still produce a complete, real verdict.

## Planned models

None currently planned — remaining milestones (13+) are frontend/infra work.

## Sports-specific intelligence (Milestone 12)

The layer that makes this a *sports* deepfake platform rather than a generic one — but built
honestly within real constraints: no sports-specific pretrained models exist off-the-shelf (no
"jersey classifier" or "stadium recognizer" you can just download), so all four are classical CV,
like Milestone 8's forensic layer.

| Detector | Technique | Scope |
|---|---|---|
| `jersey_analysis.py` | HSV color stability of the torso region (proxy: area below a detected face) across video frames | Video only (2+ frames) |
| `scene_analysis.py` | Background color-histogram distance (Bhattacharyya) across consecutive frames | Video only (2+ frames) |
| `broadcast_analysis.py` | Error Level Analysis comparing the frame's border/corner "overlay zone" (where scoreboards/logos/tickers live) against its center | Any frame |
| `crowd_analysis.py` | Tiled cosine-similarity search for duplicated crowd-texture patches (copy-pasted stadium padding) | Any frame |

Both video-only checks reuse `detection/face_detection.py`'s Haar cascade rather than
introducing a new asset. All four check *internal self-consistency* of the uploaded media —
none identify or verify against a real team's actual colors, a real stadium's actual appearance,
or a real broadcaster's actual graphics package, which would require maintained reference
databases that don't exist here.

**Honest limitations found during testing, not glossed over:**
- `crowd_analysis.py` compares tiles on a fixed, non-overlapping grid — confirmed empirically
  that a duplicated patch is caught cleanly (similarity 1.0) when it happens to land on matching
  grid positions, but a duplicate shifted by a few pixels off-grid is missed entirely. A
  shift-invariant version (sliding-window or keypoint-based, e.g. ORB/SIFT) would catch more but
  is meaningfully more expensive.
- Fixing a real bug along the way: cosine similarity on raw (non-zero-mean) pixel tiles is
  dominated by shared brightness, not texture — two *unrelated* random tiles of similar
  brightness scored ~0.95 similarity before tiles were mean-centered first.

**Deliberately out of scope** (would need external data this project doesn't have, rather than
being faked):
- **Athlete identity verification** — needs a reference-photo database of known athletes and a
  face-recognition/embedding model to compare against; there's no such database here.
- **Match context verification** (e.g. confirming a shown scoreline against real results) —
  needs a live sports-data API integration and scoreboard OCR; out of scope for a CV forensics
  module.

## Fusion weighting (updated in Milestone 12)

Three pools now, not two: DL (nominal 0.4), classical forensic — frequency, compression,
lighting, landmark, optical flow, temporal consistency (nominal 0.35, split ~0.058 each), and
sports intelligence — jersey, scene, broadcast overlay, crowd texture (nominal 0.25, split
0.0625 each). Same renormalize-across-applicable-signals behavior as Milestone 9; all pool
weights configurable via `.env` (`FUSION_DL_WEIGHT`, `FUSION_FORENSIC_WEIGHT`,
`FUSION_SPORTS_WEIGHT`).

## Video temporal extension (Milestone 11)

Extends the DL detector and forensic layer from "judge frame 0 only" to genuinely reasoning
across time for video:

- **`image_detector.py::predict_video`** — runs the same ViT classifier (Milestone 7) across up
  to 8 evenly-spaced sampled frames instead of just the first one. Its mean fake-probability
  becomes the `deep_learning` fusion input for video; fails fast (one load attempt, not one per
  frame) if the model can't be loaded.
- **`temporal_consistency` signal** — the *standard deviation* of fake-probability across those
  frames. A genuine, unedited clip of one identity should get fairly consistent classifier
  confidence frame to frame; large swings can indicate localized/frame-level tampering. Video
  only (needs 2+ analyzed frames); cascades to "unavailable" if the DL model itself couldn't
  load, since there's nothing to compute variance over.
- **`optical_flow_analysis.py`** — dense optical flow (Farneback) between consecutive frames;
  measures the spatial "roughness" (Laplacian variance) of the flow-magnitude field, the same
  idea as `frequency_analysis.py`'s spectral bumpiness applied to motion instead of pixel
  intensity. No pretrained weights — pure OpenCV, fully testable offline.

**Honest limitation, found during testing, not glossed over:** optical flow roughness responds
to *any* spatial complexity in the flow field, including entirely natural causes — a moving
subject's silhouette against a static background produces genuine flow discontinuities at its
edges via ordinary occlusion, no manipulation involved. In testing, a synthetic scene with
structured foreground objects under simple translation scored *higher* roughness than
independent random noise. We could not construct a synthetic scenario that cleanly isolates
manipulation-like discontinuity from ordinary scene structure — that would need labeled
real-vs-manipulated video data, out of scope here. Treat this as the weakest, noisiest signal
in the pipeline; it carries the same nominal fusion weight as the other forensic signals, but
real-world tuning would likely warrant weighting it down.

Fusion weighting note: with two new signal slots, the forensic pool (frequency, compression,
lighting, landmark, optical flow, temporal consistency — 6 slots) now splits the non-DL 0.5
weight six ways (~0.083 each nominal) instead of four ways; DL still gets nominal 0.5. Same
renormalize-across-applicable-signals behavior as before.

## Explainability layer (Milestone 10)

`app/services/explainability/reasoning.py` turns the fusion engine's `detector_breakdown` into
the natural-language verdict + reasons format that's actually shown to users
(`Analysis.explanation`):

```
Verdict: AUTHENTIC — no significant signs of manipulation detected

Reasons:
- Lighting and color balance on the face match the surrounding scene.
- Compression levels are consistent throughout, with no signs of splicing.
- No frequency-domain irregularities detected — the spectral profile looks natural.

Notes:
- AI deepfake classifier was unavailable for this run — verdict relies on forensic signals only.
- Facial landmark instability could not be assessed (needs 2+ frames with a detected face, or
  the landmark model was unavailable).
```

Every phrase is templated from scores the fusion engine already computed — this layer adds no
new detection logic. A signal only becomes a "reason" if its score crosses
`settings.explanation_low_threshold` / `explanation_high_threshold` (default 0.35 / 0.65);
scores in between are treated as inconclusive and omitted. Reasons are ordered by
*noteworthiness × fusion weight*, so a strong, heavily-weighted signal surfaces before a weak,
lightly-weighted one. Unavailable signals (as seen throughout this sandbox — no HF Hub/Google
storage access) become explicit "Notes" rather than silently vanishing.

The full `detector_breakdown` JSON (raw scores, weights, the technical summary, and this
rendered report) stays on the record for anyone who wants the underlying numbers, not just the
prose.
