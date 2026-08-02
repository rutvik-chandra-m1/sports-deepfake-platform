# ml/ — Data, Training, and Evaluation Tooling

Separate from `backend/` on purpose: the production FastAPI service has no reason to depend on
`datasets`, `diffusers`, or other training-only packages, and this tier has no reason to depend
on `fastapi`/`uvicorn`. Two independent virtual environments, two independent requirements files.

```
ml/
├── requirements.txt   # datasets, diffusers, torch, scikit-learn, etc. -- own .venv
├── data/               # R2: dataset acquisition
│   ├── fetch_hf_backbone.py       # general real-vs-AI-generated backbone (streamed from HF Hub)
│   ├── fetch_wikimedia_sports.py  # real sports photos (Wikimedia Commons, CC-licensed only)
│   ├── generate_synthetic.py      # synthetic sports scenes (local segmind/tiny-sd, no real subjects)
│   ├── build_manifest.py          # merges the three sources into datasets/manifest.csv with splits
│   ├── audit_dataset.py           # confound audit -- RUN THIS BEFORE TRUSTING ANY METRIC
│   └── normalize_dataset.py       # uniform transform chain + non-photograph filtering
├── eval/                # R3: evaluation harness
│   ├── run_inference.py           # scores the dataset with the REAL pipeline -- BACKEND venv
│   └── evaluate.py                # metrics, ablations, plots -- ML venv
└── train/               # R6: fine-tuning (not yet built)
```

## Setup

```bash
cd ml
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Building the R2 pilot dataset

See `docs/dataset.md` for what this produces, sourcing/licensing details, and known limitations.
Output lands in `../datasets/` (gitignored except manifest/attribution `*.csv` files):

```bash
cd data
python fetch_hf_backbone.py --per-class 350
python fetch_wikimedia_sports.py --per-category 15
python generate_synthetic.py --n 25          # slow on CPU: ~227s/image measured -- see docs/dataset.md
python build_manifest.py

# Audit BEFORE using the data. Then normalize, then re-audit to confirm.
python audit_dataset.py  --json-out ../../reports/dataset_audit.json
python normalize_dataset.py
python audit_dataset.py  --manifest manifest_normalized.csv \
                         --json-out ../../reports/dataset_audit_normalized.json
```

`manifest_normalized.csv` is the one downstream work (R3 evaluation, R6 training) should use.

## Why the audit step exists

The first build of this dataset was **perfectly separable without looking at a single pixel** —
a classifier on nothing but image width, squareness and file format scored **ROC-AUC 1.0000**.
Every "real" was ≤640px wide, every general "fake" was exactly 1024×1024, and every sports
"fake" was a 512×512 PNG while the reals were non-square JPEGs. Going straight to evaluation
would have produced a ~99%-accuracy report that measured *nothing but file dimensions*.

`audit_dataset.py` scores two tiers separately, because they mean opposite things:

- **Tier 1, container** (dimensions, aspect, squareness, format) — carries zero information
  about whether a scene was photographed or generated. Must sit near 0.50; anything higher is
  a dataset defect to fix, not a result.
- **Tier 2, content statistics** (saturation, brightness, compressibility) — generated imagery
  genuinely *does* skew more saturated, more evenly exposed and smoother. This is real but
  extremely shallow signal, so it is the **trivial baseline R3 must beat**, not a defect.

## Evaluating the pipeline (R3)

Two venvs, because inference needs the backend's torch/opencv and metrics need scikit-learn:

```bash
cd eval
../../backend/.venv/Scripts/python.exe run_inference.py     # ~0.4s/image warm
../.venv/Scripts/python.exe evaluate.py --split test
```

`run_inference.py` calls the production `analyze_frames()` directly rather than reimplementing
it, so the numbers describe the shipping system. Results: **`docs/evaluation.md`** — read that
before drawing any conclusion about detection quality. Current headline is a negative result
(test ROC-AUC 0.433, CI 0.301–0.579, below the 0.580 content-statistics baseline).

`normalize_dataset.py` is the fix for Tier 1: every image goes through an identical chain
(RGB → centre-crop square → 384×384 LANCZOS → JPEG q90), and non-photographs are dropped
(B&W newsprint scans, 1910s line drawings — neither photographed nor AI-generated, so labelling
them "real" teaches a detector nonsense). See `docs/dataset.md` for the measured before/after
and the signal cost this incurs.
