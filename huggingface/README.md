---
language:
- en
license: other
library_name: openpi
datasets:
- lerobot/libero_10
tags:
- robotics
- vision-language-action
- pi0.5
- openpi
- libero
- failure-recovery
- history-conditioning
- jax
---

# π0.5 History-Conditioned Failure Recovery

This repository contains three inference checkpoints from a controlled study of
history-conditioned failure recovery in π0.5. The policy receives the current
two-camera observation, robot state, task text, and up to eight recent
action–result records compressed into 16 history tokens. It predicts a
`10 × 7` continuous action chunk.

The accompanying code, OpenPI overlays, evaluation protocol, and raw result
summaries are available in the
[GitHub repository](https://github.com/JackyHu0314/Pi0.5-History-Conditioned-Failure-Recovery).

## Included checkpoints

| Directory | Seed | Training steps | Frozen control success | Frozen slip success |
|---|---:|---:|---:|---:|
| `checkpoints/rd-low-s21` | 21 | 1,500 | 70.0% | 80.0% |
| `checkpoints/rd-low-s22` | 22 | 1,500 | 70.0% | 80.0% |
| `checkpoints/rd-low-s23` | 23 | 1,500 | 90.0% | 90.0% |
| **Mean ± sample SD** | — | — | **76.7% ± 11.5%** | **83.3% ± 5.8%** |

Each checkpoint contains OpenPI-compatible Orbax/Zarr `params` and normalization
`assets`. Optimizer state is intentionally excluded. Seed 23 is the strongest
frozen-test run; use all three seeds when reporting aggregate results.

## What was evaluated

The frozen evaluation uses one LIBERO-10 task (task 5) and ten held-out initial
states (indices 0–9). After the policy lifts and moves the object by at least
12 mm, the slip intervention teleports it to a fixed free-joint pose, clears its
velocity, records the post-intervention result frame in history, and hides both
cameras for three policy queries (15 control steps). Control and slip rollouts
share the same reset and action prefix.

The training set contains 26 successful control/slip pairs, 52 physical recovery
windows, and 1,040 repeated recovery episodes mixed with the original LIBERO
demonstrations. Training jointly updates the history compressor and π0.5 action
expert (about 436M trainable parameters out of 3.359B total parameters) for
1,500 steps with batch size 8 on 8 × RTX 5090D 32 GB GPUs.

## Evidence boundary

Recovery distillation improves the frozen slip-success point estimate relative
to the original single-seed history model (`50.0%`), but it does **not** clearly
beat the no-history B2-zero, last-valid-frame B2-LVF, or parameter-matched C1
baselines (`80.0%` slip success each, single seed). Per-seed paired McNemar tests
against the original history model are not statistically significant
(`p = 0.25`, `0.25`, and `0.125`).

The intervention also does not increase average early re-grasp frequency or
produce a larger first-blind-action response. A linear probe can decode part of
the control/slip state from the history representation, which suggests that the
remaining bottleneck is routing history information into recovery actions.

These checkpoints therefore support a reproducible mechanism study. They do not
establish state-of-the-art performance, general failure recovery, cross-task
generalization, or real-robot capability.

## Loading and inference

1. Clone OpenPI and check out commit
   `215abfb217dbac7d5f1273282331b9b1866c0479`.
2. Apply the training/evaluation overlay from the linked GitHub repository.
3. Download this model repository and point OpenPI at one checkpoint directory:

```bash
hf download hax404/pi05-history-conditioned-failure-recovery \
  --local-dir ./pi05-history-conditioned-failure-recovery

export PI05_CHECKPOINT_DIR="$PWD/pi05-history-conditioned-failure-recovery/checkpoints/rd-low-s23"
```

The public GitHub repository documents the full paired controlled-slip protocol
and provides the evaluation runner. These checkpoints require the custom history
input fields and model overlay; they are not drop-in checkpoints for unmodified
OpenPI.

## Checkpoint format and integrity

- Format: OpenPI Orbax/Zarr directory checkpoint.
- Model dtype: bfloat16 configuration; stored tensors retain their checkpoint
  dtypes.
- Action horizon: 10; action dimension used by LIBERO: 7 (model maximum: 32).
- History: 8 records selected as `early2_recent6`, 2 learned queries per record,
  16 output tokens, hidden width 512.
- `SHA256SUMS` contains hashes for every published file.
- `manifest.json` records file counts, byte sizes, seeds, commits, and results.

## Licenses and attribution

The checkpoints contain π0.5/PaliGemma-derived parameters and are subject to the
current [Gemma Terms of Use](https://ai.google.dev/gemma/terms), including the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
The required terms and notice are included in this repository. OpenPI code and
modifications retain their Apache-2.0 attribution. The project-authored
experiment code and documentation in the linked GitHub repository use the MIT
license.

Downloading or using these checkpoints constitutes acceptance of the applicable
Gemma terms. This model repository does not replace those terms with the MIT
license used for project-authored code.

## Citation

If this artifact is useful, cite the repository URL and the exact checkpoint
seed. The project is an independent experiment built on OpenPI and LIBERO; it is
not an official Physical Intelligence release.
