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
│   └── build_manifest.py          # merges the three sources into datasets/manifest.csv with splits
├── eval/                # R3: evaluation harness (not yet built)
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
python fetch_hf_backbone.py --per-class 250
python fetch_wikimedia_sports.py --per-category 15
python generate_synthetic.py --n 25          # slow on CPU: ~227s/image measured -- see docs/dataset.md
python build_manifest.py
```
