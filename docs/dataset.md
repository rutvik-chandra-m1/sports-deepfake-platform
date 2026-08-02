# Dataset Card

> R2 pilot dataset (built 2026-08-01, rebuilt 2026-08-02 after the confound audit below).
> **Pilot scale by explicit scope decision** — meant to get the R3 evaluation harness and R4
> calibration genuinely working on real data, not to be a final-submission-scale corpus.
> Scaling up is a documented follow-up (re-run the same scripts with larger arguments), not a
> rebuild.

**Use `datasets/manifest_normalized.csv` for all evaluation and training.** The unnormalized
`manifest.csv` is retained only for provenance.

## Composition (final, post-normalization)

| Source | Class | Domain | Count | License |
|---|---|---|---:|---|
| [`ComplexDataLab/OpenFake`](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (`core`) | real | general | 218 | See dataset card |
| [`ComplexDataLab/OpenFake`](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (`core`) | fake | general | 221 | See dataset card |
| Wikimedia Commons (`ml/data/fetch_wikimedia_sports.py`) | real | sports | 40 | Per-image: public domain / CC0 only, see `datasets/sports_real/attribution.csv` |
| `segmind/tiny-sd` local generation (`ml/data/generate_synthetic.py`) | fake | sports | 25 | creativeml-openrail-m (model); images wholly synthetic, no real subject |

**Total: 504 images** (575 fetched, 71 rejected by the normalization filters below).

| Split | real | fake | total |
|---|---:|---:|---:|
| train | 155 | 163 | 318 |
| val | 65 | 48 | 113 |
| test | 38 | 35 | 73 |

### Generator diversity

**51 distinct generators** across the 246 fake images that carry generator metadata — including
DALL·E 3, Midjourney 6, FLUX (1-dev / 1.1-pro / schnell), Stable Diffusion 1.5/2.1/3.5/XL,
Ideogram 3.0, Imagen 3.0, Grok-2-image, Qwen-Image, and Playground v2.5. Recorded per-image in
the `generator` column so R3 can report per-generator performance rather than assuming uniform
difficulty. This matters: a detector that only recognises one generator's fingerprint would look
strong on a single-generator set and fail in the wild.

The 25 sports fakes are the exception — all from `segmind/tiny-sd`, since they were generated
locally. Treat any sports-domain result as generator-specific.

---

## ⚠️ Confound audit — read this before trusting any metric

**The first build of this dataset was perfectly separable without looking at a single pixel.**

`ml/data/audit_dataset.py` fits a classifier on *container properties only* — width, height,
aspect ratio, squareness, file format — deliberately excluding all image content. On a sound
dataset that should score ~0.50 ROC-AUC (chance). The first build scored **1.0000**: a single
threshold (`width > 640`) classified the general subset perfectly.

Going straight to evaluation would have produced a ~99%-accuracy report measuring **file
dimensions, not AI-generation detection** — a number that looks like a triumph and means nothing.
This is textbook shortcut learning (cf. Geirhos et al., *Shortcut Learning in Deep Neural
Networks*, 2020), and it is invisible unless explicitly tested for.

Inspecting images directly (not just counting them) surfaced label-quality problems no
count-based check would catch: the Wikimedia pull included **1930s black-and-white newsprint
scans** and a **line drawing of ancient Greek athletes from a 1910 book**, all labelled "real".
Those are neither photographs nor AI-generated images.

### The original backbone was withdrawn

`Parveshiiii/AI-vs-Real` was the first choice. Measured directly, after downloading 700 images
with shuffled sampling:

| Class | Distinct sizes across 350 images | Size |
|---|---:|---|
| real | **1** | every image exactly 178×218 |
| fake | **1** | every image exactly 1024×1024 |

Completely disjoint. The "real" half is a thumbnail scrape, the "fake" half a generation dump —
class and source perfectly correlated. **Not fixable by normalization**: equalizing sizes requires
upscaling the reals 178→384, stamping interpolation artefacts onto exactly one class, trading one
confound for another. (An earlier *unshuffled* pull from the same dataset returned reals at
640×426 — a different region of the file entirely, so the dataset is internally heterogeneous too:
any sample is unrepresentative in a way that depends on where you happen to read.)

Replaced with **OpenFake**, a published 2025 research benchmark. Its classes also differ in
resolution — cameras shoot bigger than generators emit — but **every image is above the 384px
normalization target**, so normalization only ever downsamples and no single class picks up
upscaling artefacts. That size gap is the realistic camera-vs-generator difference, not careless
construction.

`ml/data/probe_hf_dataset.py` now vets any candidate dataset **before** bulk download, and
distinguishes "needs upscaling ⇒ unfixable" from "downsample-only ⇒ fixable by normalization".

### The fix

`ml/data/normalize_dataset.py` puts every image through one identical chain
(RGB → centre-crop square → 384×384 LANCZOS → JPEG q90) and drops non-photographs. The target is
deliberately *smaller than every source*: if it matched the 512×512 synthetics, those would pass
through unresampled while real photos were downsampled, and the resampling artefact would become
a new leak pointing the other way.

Rejection filters (71 images dropped, all logged with reasons at run time):
- **too small** — would need upscaling to reach 384
- **effectively greyscale** — mean channel spread < 8 (B&W newsprint scans)
- **not a photograph** — quantized-colour entropy < 4.0 bits (line art, diagrams)

### Measured result

| Tier | Before | After | Requirement |
|---|---:|---:|---|
| **Container** — ALL | 1.0000 | **0.5000** | ~0.50 |
| **Container** — general | 1.0000 | **0.5000** | ~0.50 |
| **Container** — sports | 1.0000 | **0.5000** | ~0.50 |
| **Content-stats** — ALL | — | 0.5797 | baseline to beat |
| **Content-stats** — general | 0.7751 | **0.6071** | baseline to beat |
| **Content-stats** — sports | 0.8350 | 0.8350 | baseline to beat |
| **Duplicate leakage** | untested | **0** | zero |

Container features now carry exactly +0.0000 permutation importance — they are constant across
classes, so there is nothing left to exploit. Full JSON in `reports/dataset_audit_normalized.json`.

### How to read the three tiers

They mean different things and must not be conflated:

1. **Container** (dimensions, aspect, squareness, format) — carries zero information about whether
   a scene was photographed or generated. Must be ~0.50. Higher = dataset defect to fix, not a
   result.
2. **Content statistics** (saturation, brightness, compressibility) — generated imagery genuinely
   *does* skew more saturated, more evenly exposed and smoother. Real but very shallow signal, so
   this is **the trivial baseline R3 must beat**. R3 should report detector performance against
   this number, not against 0.50.
3. **Near-duplicate leakage** (perceptual difference-hash, robust to resize/re-encode) — verifies
   no image appears in both train and test, and no duplicate carries contradictory labels.

### Cost of the fix, stated plainly

Uniform downsampling and re-encoding **attenuates exactly the evidence `frequency_analysis.py`
and `compression_analysis.py` rely on** — high-frequency generator fingerprints and JPEG
compression history. Those two signals should be expected to underperform on this normalized set
relative to native-resolution data. Accepted because the alternative — leaving a perfectly
separable confound in place — makes every downstream number meaningless rather than merely
weaker. Uniform preprocessing is standard for detection benchmarks, and images circulating in the
real world have generally been resized and re-encoded anyway, so the normalized set is arguably
closer to the deployment condition than raw generator output.

---

## What "fake" means here — read before drawing conclusions

This is a **wholly-AI-generated-image** dataset, matching the PPT's actual framing ("Detection of
AI-Generated Sportsman Images") — **not** a deepfake/face-swap dataset. No image in the `fake`
class is a real, identifiable athlete manipulated or reenacted. The sports fakes are entirely
fictional scenes from generic prompts (e.g. "a soccer player kicking a ball on a grass field,
photorealistic") — no real person, team, league or venue named in any prompt. Visually
spot-checked: generations include convincing close-up, mid-action and wide-stadium shots.

**The evaluated detector's scope is bounded accordingly**: this is evidence about detecting
wholly-synthetic imagery, not about detecting face-swapped or reenacted footage of real athletes.
That would need consenting-subject data such as FaceForensics++, which requires a
registration/access-approval process this pass did not pursue.

## Licensing and attribution

- **OpenFake backbone** — see the dataset's own card for terms and the provenance of its real and
  generated halves. This project did not independently re-verify its construction.
- **Wikimedia Commons sports photos** — only public-domain and CC0 images were accepted (the
  fetcher also permits CC-BY/CC-BY-SA but none survived filtering). Per-image licence, source URL
  and artist are recorded in `datasets/sports_real/attribution.csv`.
- **Synthetic sports images** — no real subject, no attribution obligation. Prompt, seed and model
  recorded per-image in `datasets/sports_fake/generation_manifest.csv` for reproducibility.

## Splits

Train/val/test = 70/15/15, assigned deterministically by hashing a **group key** rather than
per-file, to avoid leakage:

- **Backbone** — grouped by filename (independent dataset entries, not a photoshoot burst).
- **Wikimedia** — grouped by Commons category, so the same photographer/event can't straddle
  train and test.
- **Synthetic** — grouped by prompt, so all images of one synthetic "scene" stay together.

Re-running `build_manifest.py` reproduces the same split exactly (pure function of the group key,
nothing randomly sampled). Verified empirically: Tier-3 audit reports zero cross-split duplicates.

## Known limitations

- **Pilot scale.** The test split is 73 images. Confidence intervals will be wide; R3 numbers are
  directional, not final accuracy claims.
- **Sports subset is small (n=65) and single-generator.** All 25 sports fakes come from
  `segmind/tiny-sd`, and its content-stats baseline is high (0.835 ± 0.131 — note the large
  standard deviation at this size). Any sports-domain conclusion is weak and generator-specific.
- **No real photographic counterfactual for the sports fakes** — there is no "same scene, but
  real" pair, so the sports real/fake distinction may partly reflect diffusion rendering
  characteristics rather than sports-specific cues. Worth an explicit ablation in R3.
- **Wikimedia sports coverage skews** toward well-photographed professional/Olympic events;
  amateur sport and broadcast screenshots are underrepresented.
- **Backbone provenance is inherited, not verified** — see OpenFake's own card.
- **Streaming shuffle is windowed.** `--shuffle-buffer` randomizes within a reservoir, not
  globally, so on a class-sorted source it improves sample diversity but does not fully mix
  classes.

## Reproducing

```bash
cd ml
python -m venv .venv && .venv\Scripts\Activate.ps1   # or source .venv/bin/activate
pip install -r requirements.txt

cd data
# 1. Vet any candidate backbone BEFORE downloading it in bulk
python probe_hf_dataset.py --dataset ComplexDataLab/OpenFake --config core --n 60

# 2. Acquire
python fetch_hf_backbone.py --per-class 250
python fetch_wikimedia_sports.py --per-category 15
python generate_synthetic.py --n 25       # slow on CPU: ~227s/image measured

# 3. Merge, normalize, and verify the confound is gone
python build_manifest.py
python audit_dataset.py --json-out ../../reports/dataset_audit.json
python normalize_dataset.py
python audit_dataset.py --manifest manifest_normalized.csv \
                        --json-out ../../reports/dataset_audit_normalized.json
```

**Scaling up**: raise `--per-class` / `--per-category` / `--n` and re-run. Increase
`--seed-start` on `generate_synthetic.py` to add synthetic images without regenerating existing
ones. Always re-run the audit afterwards — a larger sample can reintroduce confounds a smaller
one didn't have.

## Tooling notes

Two dependency issues hit while building this, fixed rather than worked around:
- `diffusers==0.35.2`'s `AutoPipelineForText2Image` fails to import under `transformers==5.14.1`
  (it eagerly imports its whole pipeline registry, including a HunyuanDiT pipeline referencing a
  class transformers 5.x removed). Fixed by importing `StableDiffusionPipeline` directly — no
  auto-detection needed since the architecture is known.
- `segmind/tiny-sd` ships its `unet/` and `vae/` as legacy pickle `.bin` weights, not
  safetensors, so `diffusers` loads them via `torch.load`'s pickle path. Pickle deserialization
  is a code-execution risk in general; accepted here because it is Segmind's own official repo and
  this is a one-time local script, not a path exposed to untrusted input.

Measured generation cost: **226.6s/image** (steps=20, guidance 7.5, 512×512, CPU-only on an Intel
i7-8665U) — ~94 minutes for 25 images, after a one-time ~5 min model download.
