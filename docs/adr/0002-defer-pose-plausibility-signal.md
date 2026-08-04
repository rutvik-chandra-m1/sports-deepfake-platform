# ADR 0002 — Defer the sports pose-plausibility signal (R8)

**Status:** Accepted
**Date:** 2026-08-04
**Supersedes:** nothing
**Related:** [ADR 0001](0001-vit-pytorch-instead-of-cnn-tensorflow.md), [`docs/evaluation.md`](../evaluation.md)

## Context

Roadmap item R8 ("sports intelligence phase 2") proposed a **pose-plausibility
signal**: run a pose estimator over detected athletes and score joint
configurations for anatomical/biomechanical implausibility, on the theory that
generative models produce subtly impossible limb geometry.

It is an appealing idea and it is the single most sports-specific detector the
project could add. It is also the one thing this repository cannot currently
tell the truth about.

## The blocking fact

The sports data is too small to measure anything:

| Split | Sports images |
|-------|---------------|
| train | 22            |
| val   | 33            |
| test  | **10** (3 real / 7 fake) |

`docs/evaluation.md` already records the consequence: the sports test row
shows ROC-AUC 0.8350 with a **95% CI of 0.000–0.556** — a confidence interval
that does not even contain the point estimate's own optimistic reading, and
which spans from "worse than a coin" to "barely better". The document's
existing instruction is blunt: *do not quote this number*.

With n=10, a signal that flips **one** image changes accuracy by 10
percentage points. Any result — improvement or regression — is
indistinguishable from noise.

## Decision

**Do not implement the pose-plausibility signal yet, and do not wire any
unmeasured signal into the fusion engine.**

This is a deliberate, recorded decision, not an oversight or an unfinished
task.

## Why not "implement it anyway, it can only help"

Because it can hurt, and we would not be able to see it.

1. **Fusion weights are renormalised across applicable signals.** Adding a
   signal takes weight away from `trained_probe` — the only signal in this
   system with a measured above-chance result (0.7534 AUC standalone). An
   unvalidated signal would dilute the one thing demonstrably working.
2. **It would be inherited by every verdict**, including the ~265 non-sports
   test images where a pose estimator on a non-sports photograph produces
   whatever it produces.
3. **It would look like progress.** A new row in the signal breakdown, a
   heavier `detector_breakdown`, a more impressive-sounding architecture — and
   no evidence behind any of it. That is exactly the failure mode this project
   spent R3 correcting, when the platform emitted confident verdicts at
   **0.4331 AUC, below chance**.

The project's own recorded conclusion after R3/R6 stands: *the highest-value
single action for detection quality is more training data, not more
architecture*.

## Gate for revisiting

R8 becomes implementable when **all** of the following hold:

1. **≥200 sports test images**, at roughly balanced real/fake — enough that a
   bootstrap 95% CI on sports ROC-AUC is narrower than ±0.10.
2. The expanded sports set **passes `ml/data/audit_dataset.py`** at both the
   container tier and the content-statistics tier. This is not optional: an
   earlier dataset in this project reached container-only AUC **1.0000** on
   image width alone, and a sports set assembled from one real source and one
   generator would reproduce that confound exactly.
3. Fakes come from **≥2 distinct generators**, so the signal cannot learn one
   model's artefact signature and be reported as "pose plausibility".

Existing tooling covers acquisition: `ml/data/fetch_wikimedia_sports.py` for
real imagery and `ml/data/generate_synthetic.py` for fakes. The binding
constraint is CPU generation throughput, not code — the current set contains 7
synthetic sports images for that reason.

## Consequences

- The platform ships with **no sports-specific pose signal**. The existing
  sports-related signals from Milestone 12 remain, at their existing low
  weight.
- The measured, quotable performance of the system is unchanged: **ROC-AUC
  0.7402 (95% CI 0.680–0.796), accuracy 71.3% on 275 held-out images**, still
  general-purpose rather than sports-specific.
- Every user-facing surface already states this. The generated report
  (`app/services/reporting.py`) carries the limitation *"Sports-specific
  performance is effectively unmeasured (only 10 sports images in the test
  set)"* on page one, before the verdict.
- If a reviewer asks "where is the sports intelligence?", the answer is this
  document: the capability is specified, its blocker is quantified, and the
  gate to unblock it is written down.
