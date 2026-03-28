# PDQ Evasion Attack Plan

## Current Status

- Existing repo PDQ support is present, but it is black-box only.
- `utils/pdq_torch.py` uses the exact `pdqhash` oracle and does not expose a differentiable surrogate.
- `fdeph_eval/attacks/pdq_evasion_steps.py` is an SPSA attack, and `fdeph_eval/attacks/pdq_evasion_steps_patchtest.py` already suggests the key fix: perturbed PDQ queries must be thresholded with each image's current median, not the original image median.
- The new isolated work for this phase lives under `fdeph_eval/attacks/pdq/` so current code paths remain unchanged.

## Research Direction

Two attack tracks are worth testing:

1. Black-box SPSA against the exact PDQ oracle.
2. White-box surrogate optimization through a PDQ-style differentiable pipeline.

The rationale is straightforward:

- Prokos et al. describe PDQ as a classical perceptual hash built from luma preprocessing and DCT-domain quantization, which makes a surrogate attack technically plausible even if the production implementation is not directly differentiable.
- Madden et al. evaluate PDQ under constrained black-box adversarial settings. That leaves room for stronger white-box surrogate testing.
- The official Python bindings expose both `compute(...)` and `compute_float(...)`, which makes exact hard-hash validation and pre-binarization inspection possible.

Sources:

- USENIX Security 2023, “Squint Hard Enough: Attacking Perceptual Hashing”: https://www.usenix.org/system/files/sec23summer_146-prokos-prepub.pdf
- PDQ Python bindings / ThreatExchange wrapper behavior: local `pdqhash` package plus repo usage via `utils/pdq_torch.py`
- Jordan Madden et al., “Assessing the Adversarial Security of Perceptual Hashing Algorithms”: https://dblp.org/rec/journals/corr/abs-2406-00918

## New Files

- `pdq_surrogate.py`: differentiable PDQ-style surrogate.
- `pdq_evasion_blackbox_spsa.py`: exact-oracle SPSA attack with current-median queries.
- `pdq_evasion_whitebox_surrogate.py`: white-box attack using the surrogate, but exact oracle for success checks.
- `prepare_subset_pdq.py`: creates a 10-image subset with filenames suffixed `_pdq`.
- `run_subset_attacks_pdq.py`: sequential pilot runner for both attacks.

## Imagenette Subset Choice

- This repo's `datasets/imagenette.py` and `experiments/make_inputs_sample.py` both indicate Imagenette is expected under `data/imagenette2-320/`.
- For evaluation, the correct raw-image source is `data/imagenette2-320/val`.
- The subset builder now defaults to `data/imagenette2-320/val` and picks 10 images in a class-balanced way: one image from each of Imagenette's 10 classes.
- The output subset remains flat under `inputs/pdq_subset_10`, and every copied filename ends with `_pdq`.

## Suggested Run Flow

Build the 10-image Imagenette subset:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/prepare_subset_pdq.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/pdq_subset_10 \
  --n 10 \
  --seed 42 \
  --clear
```

Then run both pilot attacks:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_subset_attacks_pdq.py \
  --source ./inputs/pdq_subset_10 \
  --threshold 0.08 \
  --blackbox_steps 300 \
  --whitebox_steps 300
```

## What To Look For

- Any image that reaches normalized Hamming distance `>= 0.08` is an early positive signal.
- Compare black-box and white-box step efficiency.
- If white-box succeeds where SPSA stalls, the surrogate is useful even if imperfect.
- If both fail on the 10-image pilot, the next step is tuning `temperature`, SPSA sample count, and visual-loss weight before scaling up.
