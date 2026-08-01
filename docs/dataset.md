# Dataset Card

> R2 pilot dataset (2026-08-01). This is a **pilot-scale** dataset (~150-300 images/class,
> per an explicit scope decision — see `docs/milestones.md`), meant to get the R3 evaluation
> harness and R4 calibration genuinely working end-to-end on real data. It is not the ≥2,000
> image/class target the original roadmap described for a final submission — scaling up is a
> documented follow-up (re-run the same scripts with higher `--per-class`/`--n`/`--per-category`
> values), not a rebuild.

## Composition

Three sources, merged by `ml/data/build_manifest.py` into `datasets/manifest.csv`:

| Source | Class | Domain | Count | License |
|---|---|---|---:|---|
| [`Parveshiiii/AI-vs-Real`](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) (HF Hub) | real | general | 250 | MIT |
| [`Parveshiiii/AI-vs-Real`](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) (HF Hub) | fake | general | 250 | MIT |
| Wikimedia Commons (`ml/data/fetch_wikimedia_sports.py`) | real | sports | 50 | Per-image: 33 public domain, 17 CC0 (see `datasets/sports_real/attribution.csv`) |
| `segmind/tiny-sd` local generation (`ml/data/generate_synthetic.py`) | fake | sports | 25 | creativeml-openrail-m (model license) — wholly synthetic images, no real subject |

**Total: 575 images.** Final train/val/test split (from the actual `build_manifest.py` run, hash-based per the grouping rules below, so re-running reproduces it exactly):

| Split | real | fake | total |
|---|---:|---:|---:|
| train | 177 | 186 | 363 |
| val | 84 | 43 | 127 |
| test | 39 | 46 | 85 |

The val/test real:fake ratio isn't perfectly even (e.g. val skews real-heavy, test skews fake-heavy) — an expected consequence of hash-based grouping with a small number of groups at pilot scale (7 Wikimedia categories, 22 synthetic prompts, 2 backbone classes), not a bug. Don't read too much into per-split class balance until R2 is scaled up.

**Real counts, not the requested per-category targets — read before assuming a bug.**
Wikimedia yielded 50 of a nominal 105 (15 × 7 categories): `Category:Football players` (8),
`Category:Basketball players` (3), `Category:Tennis players` (6), `Category:Sprinters` (15),
`Category:Swimmers` (6), `Category:Volleyball players` (6), `Category:Cricketers` (6).
`Category:Track and field athletes` was tried first and returned zero — it turned out to contain
only subcategories, no files directly attached; `Category:Sprinters` was substituted after
verifying it has real file members. The shortfall in the other categories is the candidate-pool
filter (license + format + minimum-size checks) removing more candidates than expected, not a
script defect — every image that *did* pass is correctly attributed in `attribution.csv`.

**Model swap, also read before assuming a bug.** The original plan was `stabilityai/sd-turbo`.
Checked against real Hugging Face file-size metadata (not the repo's aggregate "storage used"
figure, which includes ONNX/OpenVINO/single-file exports never actually downloaded by
`diffusers.from_pretrained`): sd-turbo's diffusers-format weights are ~2.5GB (fp16) to ~4.9GB
(fp32) — projected at 5-9 hours on this machine's connection. `segmind/tiny-sd` — an
architecturally distilled SD1.5 variant (fewer UNet blocks, not just lower precision) — is ~1GB;
the actual download took ~5 minutes (this connection's throughput varies a lot run to run, see
`docs/installation.md`), far under the earlier ~2h projection. Swapped before starting any bulk
generation.

Two other things hit during setup, neither a data-quality problem:
- `diffusers==0.35.2`'s `AutoPipelineForText2Image` failed to import at all under
  `transformers==5.14.1` (eagerly imports its whole pipeline registry, including an unrelated
  HunyuanDiT pipeline that references a class transformers 5.x removed) — worked around by
  importing `StableDiffusionPipeline` directly in `ml/data/generate_synthetic.py`, which needs no
  auto-detection since the model architecture was already known.
- `segmind/tiny-sd`'s `unet/` and `vae/` components ship only as legacy pickle (`.bin`) weights,
  not `.safetensors` (unlike `text_encoder/`) — `diffusers` logged "Defaulting to unsafe
  serialization" and loaded them via `torch.load`'s pickle path. Pickle deserialization from an
  untrusted source is a real code-execution risk in general; accepted here because this is
  Segmind's own official repo (a known AI company, not a third-party re-upload) and this is a
  one-time local generation script, not something that runs against untrusted input. Worth
  revisiting if this model is ever loaded in a context with a weaker trust boundary.

Measured generation cost: 25 images averaged 226.6s/image (steps=20, guidance_scale=7.5, 512x512,
CPU-only on an Intel i7-8665U) — about 94 minutes total for the batch, after a one-time ~5 min
model load. Scaling this supplement up later costs roughly 3.8 minutes of CPU time per additional
image, not counting the (now-cached) model download.

**Why this split:** the general backbone (`Parveshiiii/AI-vs-Real`) gets real evaluation numbers
fast with zero generation/scraping time. The sports-specific supplement directly targets the gap
flagged in the original engineering review — the platform's DL detector is trained on face
imagery, not the sports action/crowd/broadcast content this project actually targets — so R3's
evaluation harness can report accuracy separately for `domain=general` vs `domain=sports` and
make that mismatch visible with numbers instead of a documented guess.

## What "fake" means here — read before drawing conclusions

This is a **wholly-AI-generated-image** dataset, matching the PPT's actual framing ("Detection
of AI-Generated Sportsman Images") — not a deepfake/face-swap dataset. No image in the `fake`
class depicts a real, identifiable athlete manipulated or reenacted. The sports-domain fake
images are entirely fictional scenes generated by `segmind/tiny-sd` from generic prompts
(e.g. "a soccer player kicking a ball on a grass field, photorealistic") — no real person, team,
league, or venue is named in any prompt. Visually spot-checked (not just counted): generations
include convincing close-up, mid-action, and wide-stadium shots across the prompt set — see
`datasets/sports_fake/*.png`. The general-domain fake images (from the HF backbone)
were not generated by this project and their exact provenance is whatever
`Parveshiiii/AI-vs-Real`'s own construction process used (see that dataset's card).

**This means the trained/evaluated detector's real-world scope is bounded accordingly**: it is
evidence about detecting wholly-synthetic sportsman imagery, not about detecting face-swapped or
reenacted footage of real athletes (that would need consenting-subject data like the original
FaceForensics++, which requires a registration/access-approval process this pass did not pursue
— see `docs/models.md`).

## Licensing and attribution

- **HF backbone**: MIT-licensed dataset, no per-image attribution required.
- **Wikimedia Commons real photos**: every file's exact license (CC0, Public Domain, CC-BY, or
  CC-BY-SA — no other license was accepted) and required attribution is recorded per-image in
  `datasets/sports_real/attribution.csv`, generated at download time by
  `ml/data/fetch_wikimedia_sports.py`. CC-BY/CC-BY-SA images require attribution to the listed
  artist if this dataset or derived images are redistributed; CC0/PD images do not.
- **Synthetic sports images**: no real subject, no attribution obligation. Generation prompt,
  seed, and model are recorded per-image in `datasets/sports_fake/generation_manifest.csv` for
  reproducibility.

## Splits

Train/val/test = 70/15/15, assigned deterministically by a hash of a **group key** rather than
per-file, to avoid the leakage the roadmap flagged as a risk:

- Backbone images: grouped by filename (these are already independent, pre-shuffled dataset
  entries, not a photoshoot burst — per-file grouping doesn't leak here).
- Wikimedia images: grouped by Commons category (e.g. all `Category:Swimmers` photos land in the
  same split), so the same photographer/event isn't split across train and test.
- Synthetic images: grouped by prompt (all images generated from one prompt template — i.e. one
  synthetic "scene" — land in the same split).

Re-running `build_manifest.py` on the same source manifests reproduces the same split (the hash
is a pure function of the group key, nothing is randomly sampled).

## Known limitations

- **Pilot scale.** Numbers from R3 on this dataset are directionally useful, not a final
  accuracy claim — expect wide confidence intervals at this sample size.
- **Sports-domain fake images have no real photographic counterfactual** — there's no "the same
  scene, but real" pair, so the fake/real distinction the detector learns for the sports subset
  may partly reflect diffusion-model rendering artifacts (composition, hands, text-like blur)
  rather than sports-specific manipulation cues. Worth an explicit ablation in R3.
- **Wikimedia sports photos skew toward well-photographed professional/Olympic-level events** —
  Commons' CC-licensed sports coverage is not a representative sample of all sports photography
  (e.g. amateur/local sports, broadcast screenshots are underrepresented).
- **General backbone provenance is inherited, not verified** — this project did not independently
  confirm how `Parveshiiii/AI-vs-Real`'s images were sourced or generated; see that dataset's own
  card for its methodology.

## Reproducing

Exact commands used to produce the numbers in this card:

```bash
cd ml
python -m venv .venv && .venv\Scripts\Activate.ps1   # or source .venv/bin/activate
pip install -r requirements.txt

cd data
python fetch_hf_backbone.py --per-class 250 --out ../../datasets/backbone
python fetch_wikimedia_sports.py --per-category 15 --out ../../datasets/sports_real
python generate_synthetic.py --n 25 --out ../../datasets/sports_fake
python build_manifest.py --datasets-dir ../../datasets --out ../../datasets/manifest.csv
```

**Scaling up** (documented follow-up, not a rebuild): re-run any of the three fetch/generate
scripts with a higher `--per-class` / `--per-category` / `--n`, then re-run `build_manifest.py`.
Existing files aren't deleted, so `fetch_hf_backbone.py` and `generate_synthetic.py` (which
index/seed sequentially from 0 / `--seed-start`) will overwrite and extend in place; increase
`--seed-start` for `generate_synthetic.py` to add new images without regenerating existing ones.
`fetch_wikimedia_sports.py` re-fetches its whole category list each run (cheap — the API calls are
fast, only the actual image downloads are slower), overwriting `attribution.csv`.
