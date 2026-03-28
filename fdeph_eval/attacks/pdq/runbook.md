# PDQ Linux Runbook

This runbook covers the isolated PDQ workflow under `fdeph_eval/attacks/pdq/`.
It is written for Linux execution from the repository root.

## Scope

These commands use only the new isolated PDQ files and do not depend on the old PDQ attack scripts.

Files used:
- `fdeph_eval/attacks/pdq/prepare_subset_pdq.py`
- `fdeph_eval/attacks/pdq/pdq_evasion_blackbox_spsa.py`
- `fdeph_eval/attacks/pdq/pdq_evasion_whitebox_surrogate.py`
- `fdeph_eval/attacks/pdq/run_subset_attacks_pdq.py`
- `fdeph_eval/analysis/pdq_attack_visualization.ipynb`

## 0. Check And Install PDQ First

Do not assume `pdqhash` is installed.

### 0A. Check whether `pdqhash` is available

```bash
python -c "import pdqhash; print('pdqhash ok')"
```

If that prints `pdqhash ok`, continue to Section 1.

If it fails with `ModuleNotFoundError`, install it.

### 0B. Install `pdqhash`

Preferred:

```bash
python -m pip install --user pdqhash
```

If you are inside a virtual environment:

```bash
python -m pip install pdqhash
```

### 0C. Verify installation

```bash
python -c "import pdqhash; print('pdqhash ok')"
python -c "import pdqhash, inspect; print(pdqhash.__file__)"
```

If installation fails because `pip` is old, update it and retry:

```bash
python -m pip install --upgrade pip
python -m pip install --user pdqhash
```

## Important Note About Full-Folder Runs

Yes, the current isolated code can run on `inputs_500` or any other specified input folder.

But both attack scripts currently default to:
- `--sample_limit 10`

So for any real run beyond the 10-image subset, you must explicitly pass one of:
- `--sample_limit 500`
- `--sample_limit 10000000`
- or another desired limit

If you forget that flag, the script will silently process only 10 images.

## Assumptions

- You are in the repo root.
- Imagenette raw images are present under:
  `./data/imagenette2-320/val`
- `PYTHONPATH=.` is used for all runs.

## 1. Environment Sanity Check

```bash
pwd
python --version
python -c "import pdqhash; print('pdqhash ok')"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 2. Clean Previous PDQ Pilot Outputs

Run this before a fresh pilot:

```bash
rm -rf ./inputs/pdq_subset_10
rm -rf ./evasion_attack_outputs_pdq_blackbox_smoketest
rm -rf ./evasion_attack_outputs_pdq_blackbox_T0.08_subset10
rm -rf ./evasion_attack_outputs_pdq_whitebox_smoketest
rm -rf ./evasion_attack_outputs_pdq_whitebox_T0.08_subset10
rm -f ./logs/attack_steps_pdq_blackbox_smoketest.csv
rm -f ./logs/attack_steps_pdq_blackbox_subset10_T0.08.csv
rm -f ./logs/attack_steps_pdq_whitebox_smoketest.csv
rm -f ./logs/attack_steps_pdq_whitebox_subset10_T0.08.csv
```

## 3. Build the 10-Image PDQ Subset From Imagenette Val

Default recommended mode: one image per Imagenette class.

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/prepare_subset_pdq.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/pdq_subset_10 \
  --n 10 \
  --seed 42 \
  --mode balanced_imagenette \
  --clear
```

Verify the subset:

```bash
ls -lh ./inputs/pdq_subset_10
find ./inputs/pdq_subset_10 -type f | wc -l
```

Expected: `10` images, each filename ending in `_pdq`.

## 4. Short Smoke Test

Use this first to confirm the environment is correct.

### 4A. Black-box smoke test

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_blackbox_spsa.py \
  --source ./inputs/pdq_subset_10 \
  --sample_limit 2 \
  --max_steps 5 \
  --threads 1 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_blackbox_smoketest.csv \
  --output_folder ./evasion_attack_outputs_pdq_blackbox_smoketest
```

### 4B. White-box smoke test

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_whitebox_surrogate.py \
  --source ./inputs/pdq_subset_10 \
  --sample_limit 2 \
  --max_steps 5 \
  --threads 1 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_whitebox_smoketest.csv \
  --output_folder ./evasion_attack_outputs_pdq_whitebox_smoketest
```

### 4C. Smoke-test sanity check

```bash
PYTHONPATH=. python experiments/sanity_check.py ./logs/attack_steps_pdq_blackbox_smoketest.csv
PYTHONPATH=. python experiments/sanity_check.py ./logs/attack_steps_pdq_whitebox_smoketest.csv
```

## 5. Main 10-Image Pilot Run

This runs both isolated attacks sequentially on the full subset of 10 images.

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_subset_attacks_pdq.py \
  --source ./inputs/pdq_subset_10 \
  --threshold 0.08 \
  --blackbox_steps 300 \
  --whitebox_steps 300
```

Outputs created:
- `./logs/attack_steps_pdq_blackbox_subset10_T0.08.csv`
- `./logs/attack_steps_pdq_whitebox_subset10_T0.08.csv`
- `./evasion_attack_outputs_pdq_blackbox_T0.08_subset10/`
- `./evasion_attack_outputs_pdq_whitebox_T0.08_subset10/`

Sanity check:

```bash
PYTHONPATH=. python experiments/sanity_check.py ./logs/attack_steps_pdq_blackbox_subset10_T0.08.csv
PYTHONPATH=. python experiments/sanity_check.py ./logs/attack_steps_pdq_whitebox_subset10_T0.08.csv
```

## 6. Alternate Threshold Pilot Runs

### 6A. T = 0.10

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_subset_attacks_pdq.py \
  --source ./inputs/pdq_subset_10 \
  --threshold 0.10 \
  --blackbox_steps 300 \
  --whitebox_steps 300
```

### 6B. T = 0.12

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_subset_attacks_pdq.py \
  --source ./inputs/pdq_subset_10 \
  --threshold 0.12 \
  --blackbox_steps 300 \
  --whitebox_steps 300
```

### 6C. T = 0.30

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_subset_attacks_pdq.py \
  --source ./inputs/pdq_subset_10 \
  --threshold 0.30 \
  --blackbox_steps 300 \
  --whitebox_steps 300
```

## 7. Run Only One Attack Variant

### 7A. Black-box only on the full subset

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_blackbox_spsa.py \
  --source ./inputs/pdq_subset_10 \
  --sample_limit 10 \
  --max_steps 300 \
  --threads 1 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_blackbox_subset10_T0.08.csv \
  --output_folder ./evasion_attack_outputs_pdq_blackbox_T0.08_subset10
```

### 7B. White-box only on the full subset

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_whitebox_surrogate.py \
  --source ./inputs/pdq_subset_10 \
  --sample_limit 10 \
  --max_steps 300 \
  --threads 2 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_whitebox_subset10_T0.08.csv \
  --output_folder ./evasion_attack_outputs_pdq_whitebox_T0.08_subset10
```

## 8. Edge-Only Scenarios

### 8A. Black-box edge-only

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_blackbox_spsa.py \
  --source ./inputs/pdq_subset_10 \
  --sample_limit 10 \
  --max_steps 300 \
  --threads 1 \
  --edges_only \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_blackbox_edges_subset10_T0.08.csv \
  --output_folder ./evasion_attack_outputs_pdq_blackbox_edges_T0.08_subset10
```

### 8B. White-box edge-only

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_whitebox_surrogate.py \
  --source ./inputs/pdq_subset_10 \
  --sample_limit 10 \
  --max_steps 300 \
  --threads 2 \
  --edges_only \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_whitebox_edges_subset10_T0.08.csv \
  --output_folder ./evasion_attack_outputs_pdq_whitebox_edges_T0.08_subset10
```

## 9. Longer Subset Runs If Early Signal Appears

If the 300-step run shows movement but no success, extend only the step budget first.

### 9A. Longer black-box run

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_blackbox_spsa.py \
  --source ./inputs/pdq_subset_10 \
  --sample_limit 10 \
  --max_steps 1000 \
  --threads 1 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_blackbox_subset10_T0.08_long.csv \
  --output_folder ./evasion_attack_outputs_pdq_blackbox_T0.08_subset10_long
```

### 9B. Longer white-box run

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_whitebox_surrogate.py \
  --source ./inputs/pdq_subset_10 \
  --sample_limit 10 \
  --max_steps 1000 \
  --threads 2 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_whitebox_subset10_T0.08_long.csv \
  --output_folder ./evasion_attack_outputs_pdq_whitebox_T0.08_subset10_long
```

## 10. Full-Folder Attack on `inputs_500` or Another Folder

This is supported by the current code.

You must set:
- `--source` to the folder you want
- `--sample_limit` to the real number of images to process

### 10A. Build `inputs_500` from Imagenette val using the existing repo sampler

```bash
PYTHONPATH=. python experiments/make_inputs_sample.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/inputs_500 \
  --n 500 \
  --seed 42 \
  --clear
```

### 10B. Black-box on `inputs_500`

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_blackbox_spsa.py \
  --source ./inputs/inputs_500 \
  --sample_limit 500 \
  --max_steps 300 \
  --threads 1 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_blackbox_inputs500_T0.08.csv \
  --output_folder ./evasion_attack_outputs_pdq_blackbox_inputs500_T0.08
```

### 10C. White-box on `inputs_500`

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_whitebox_surrogate.py \
  --source ./inputs/inputs_500 \
  --sample_limit 500 \
  --max_steps 300 \
  --threads 2 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_whitebox_inputs500_T0.08.csv \
  --output_folder ./evasion_attack_outputs_pdq_whitebox_inputs500_T0.08
```

### 10D. Run on any other folder

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_evasion_whitebox_surrogate.py \
  --source /absolute/path/to/your/input_folder \
  --sample_limit 10000000 \
  --max_steps 300 \
  --threads 2 \
  --hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_customfolder_T0.08.csv \
  --output_folder ./evasion_attack_outputs_pdq_customfolder_T0.08
```

## 11. Visualization

Open the notebook:

```bash
jupyter notebook fdeph_eval/analysis/pdq_attack_visualization.ipynb
```

Before running cells, set in the config cell:
- `ATTACK_VARIANT = 'whitebox'` or `'blackbox'`
- `THRESHOLD = '0.08'`
- If needed, update `LOG_PATH` and `OUT_DIR` to point at the run you want to visualize.

## 12. Recommended Execution Sequence

Run in this order:

1. Check whether `pdqhash` is installed.
2. Install `pdqhash` if missing.
3. Run the environment sanity check.
4. Clean previous PDQ pilot outputs.
5. Build `inputs/pdq_subset_10` from `data/imagenette2-320/val`.
6. Run black-box and white-box smoke tests.
7. Run the 10-image subset pilot at `T=0.08`.
8. If there is useful movement, run longer subset attacks.
9. If the subset shows success, scale to `inputs_500`.
10. Only after that, try alternate thresholds `0.10`, `0.12`, and `0.30`.
11. Generate visualizations from the successful run folder.

## 13. Interpreting Early Results

- If black-box moves only to `0.0078125` or `0.015625` and stalls, SPSA likely needs more steps or more samples.
- If white-box consistently moves farther than black-box, the surrogate is informative even if it is still below threshold.
- If either method reaches `>= 0.08` on the 10-image subset, proceed to `inputs_500` immediately.
- If neither method moves meaningfully after 1000 steps on the subset, tune hyperparameters before scaling up.

## 14. Current Default Hyperparameters

Black-box:
- `lr=0.02`
- `c_spsa=0.02`
- `spsa_samples=4`
- `max_steps=300`
- `ssim_weight=1.0`
- `threads=1`

White-box:
- `learning_rate=0.002`
- `temperature=20.0`
- `max_steps=300`
- `ssim_weight=5.0`
- `threads=2`

## 15. Common Mistakes To Avoid

- Do not forget `PYTHONPATH=.`.
- Do not forget `--sample_limit` for full-folder runs.
- Do not point the subset builder at `train` if you want a held-out evaluation subset.
- Do not reuse old logs when comparing runs; use fresh CSV paths or delete the old ones first.
- Do not expect the visualization notebook to show anything unless the configured output folder contains saved adversarial images.
