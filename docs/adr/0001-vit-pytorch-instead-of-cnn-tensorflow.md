# ADR 0001 — Vision Transformer + PyTorch instead of CNN + TensorFlow

**Status:** Accepted · **Date:** 2026-08-02

> **Read this before the viva.** The PPT commits to a CNN implemented in TensorFlow. The code
> uses a Vision Transformer in PyTorch. That divergence is deliberate and defensible, but it will
> be noticed, so the argument is written down here rather than improvised on the day.

## Context

The Phase-I PPT (BCS685) specifies:

- **CNN-based classification**, with a literature survey citing ResNet, XceptionNet and
  "Convolutional Traces"
- **TensorFlow** as the framework
- **Metadata provenance verification (C2PA)** as an objective
- Scope: **images** of sportsmen

The implementation diverged on the first two. The third was missing entirely until R5 and is now
implemented (`docs/models.md`). The fourth is unchanged — images remain the evaluated scope, and
the video path is explicitly reported as unevaluated.

## Decision

Use a **Vision Transformer (ViT-B/16) in PyTorch**, as a frozen backbone with a trained linear
head, rather than training a CNN in TensorFlow.

## Rationale

### 1. No GPU was available

The development machine is an Intel i7-8665U with **no CUDA device**. Training a CNN from scratch,
or full fine-tuning of any deep model, is not practical on CPU within the project timeframe.

A **frozen backbone + linear probe** needs exactly one forward pass per image (measured: 0.42s),
after which the head trains in milliseconds. This is what made a *measured, held-out* result
achievable at all. Under the alternative, the project would most likely have shipped an untrained
or barely-trained CNN with no credible evaluation — which is precisely the failure the
engineering review identified in the original codebase.

### 2. The pretrained ecosystem is PyTorch-first

The chosen starting point — a ViT already fine-tuned for deepfake classification — is distributed
via Hugging Face `transformers`, which as of v5 is **PyTorch-only** (TensorFlow support was
removed, not deprecated). Using TensorFlow would have meant abandoning pretrained deepfake weights
and starting from ImageNet or scratch, which returns to problem (1).

### 3. ViT is not a downgrade from CNN for this task

Vision Transformers are the current standard for image classification and match or exceed CNNs on
detection benchmarks. The PPT's CNN references (2018–2020) predate ViT's adoption. Choosing ViT is
moving *with* the literature, not away from it.

### 4. The measurement supports the choice

The decision is backed by a number, not a preference. The frozen ViT trunk's features, with a
trained head, reach **ROC-AUC 0.7534** on a held-out test set (n=275) — against **0.491 (chance)**
for the stock face-tuned head it replaced. See `docs/evaluation.md`.

## Consequences

**Positive**

- A real, held-out, reproducible detection result exists — the project's central deliverable.
- Transfer learning is genuinely demonstrated (frozen backbone, trained head, hyperparameters
  chosen by cross-validation *inside* the training split).
- CPU-tractable: the whole train → evaluate loop reruns in minutes.

**Negative — stated plainly**

- **The PPT is now inaccurate on two points** and must either be revised or defended verbally.
- **No CNN baseline exists.** A ViT-vs-CNN comparison would have strengthened the report and is
  the single best remaining addition (see below).
- Linear probing is weaker than full fine-tuning; some accuracy is left on the table.

## What to do about the divergence

Two acceptable resolutions — **agree one with the guide**:

1. **Revise the PPT** to say Vision Transformer / PyTorch, citing this ADR. Most honest, and
   matches what was actually built and measured.
2. **Keep the PPT and add a CNN baseline.** Train a small CNN (or ResNet-18 linear probe) on the
   same splits and report both. This satisfies the CNN claim *and* produces a genuinely valuable
   comparison. Cost: a few hours, mostly CPU time — the dataset, splits and evaluation harness
   already exist, so only a training script is new.

**Option 2 is the stronger submission** if time allows: "we tried both and here is the
comparison" is a better answer than either document alone.

## Anticipated viva questions

**"Your PPT says CNN and TensorFlow. Why is this a ViT in PyTorch?"**
No GPU was available, so training a CNN from scratch was not viable; a frozen pretrained backbone
with a trained head was. `transformers` v5 is PyTorch-only, so the pretrained deepfake weights we
build on are only reachable through PyTorch. ViT also matches or beats CNNs on this task in
current literature. The choice is recorded in ADR 0001 with the measured justification.

**"Is a linear probe really deep learning / transfer learning?"**
Yes — transfer learning is precisely reusing representations learned on one task for another.
Freezing the backbone and training a new head is the textbook linear-probe protocol used to
measure what a pretrained representation encodes. The alternative (full fine-tuning) differs in
degree, not in kind, and was infeasible without a GPU.

**"What accuracy did you achieve?"**
71.3% accuracy, ROC-AUC 0.7402 (95% CI 0.680–0.796) on a held-out test set of 275 images —
against a 53.8% majority-class baseline and a 0.5458 content-statistics baseline. And the honest
part: the original hand-weighted pipeline measured **0.4331, below chance**, which is why the
evaluation harness was built before anything else was trusted.

## Related

- `docs/evaluation.md` — the measurements
- `docs/models.md` — architecture and limitations
- `docs/dataset.md` — the confound audit that made the numbers trustworthy
