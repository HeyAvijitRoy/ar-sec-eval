# PDQ Collision Runbook

This runbook covers the isolated PDQ collision workflow under `fdeph_eval/attacks/pdq/`.
It is written for Linux execution from the repository root.

Goal:
- try exact targeted PDQ collision first
- if exact collision does not succeed, retry as thresholded target match

Files used:
- `fdeph_eval/attacks/pdq/prepare_subset_pdq.py`
- `fdeph_eval/attacks/pdq/compute_dataset_hashes_pdq.py`
- `fdeph_eval/attacks/pdq/pdq_collision_whitebox_surrogate.py`
- `fdeph_eval/attacks/pdq/run_pdq_collision_subset.py`

## 0. Check And Install PDQ First

Do not assume `pdqhash` is installed.

### 0A. Check whether `pdqhash` is available

```bash
python -c "import pdqhash; print('pdqhash ok')"
```

If that prints `pdqhash ok`, continue.

If it fails with `ModuleNotFoundError`, install it:

```bash
python -m pip install --user pdqhash
```

If you are inside a virtual environment:

```bash
python -m pip install pdqhash
```

Verify:

```bash
python -c "import pdqhash; print('pdqhash ok')"
python -c "import pdqhash, inspect; print(pdqhash.__file__)"
```

## 1. Environment Sanity Check

```bash
pwd
python --version
python -c "import pdqhash; print('pdqhash ok')"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 2. Assumptions

- You are in the repo root.
- Source images are available under `./inputs/pdq_subset_10` or another chosen folder.
- Target images are available under `./data/imagenette2-320/train` or another chosen folder.
- Commands below use `PYTHONPATH=.` for consistency.

## 3. Optional Cleanup For A Fresh Collision Pilot

```bash
rm -rf ./collision_attack_outputs_pdq_subset
rm -rf ./collision_attack_outputs_pdq_exact_smoketest
rm -rf ./collision_attack_outputs_pdq_threshold_smoketest
rm -f ./logs/attack_steps_pdq_collision_subset.csv
rm -f ./logs/attack_steps_pdq_collision_exact_smoketest.csv
rm -f ./logs/attack_steps_pdq_collision_threshold_smoketest.csv
rm -f ./dataset_hashes/imagenette_train_pdq_hashes.csv
```

## 4. Build Or Refresh The 10-Image PDQ Source Subset

Recommended for a small pilot:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/prepare_subset_pdq.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/pdq_subset_10 \
  --n 10 \
  --seed 42 \
  --mode balanced_imagenette \
  --clear
```

Verify:

```bash
ls -lh ./inputs/pdq_subset_10
find ./inputs/pdq_subset_10 -type f | wc -l
```

## 5. Build The PDQ Target Hashset

This computes hashes for the target pool that collision targets will be selected from.

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/compute_dataset_hashes_pdq.py \
  --source ./data/imagenette2-320/train \
  --output_path ./dataset_hashes/imagenette_train_pdq_hashes.csv
```

Verify:

```bash
ls -lh ./dataset_hashes/imagenette_train_pdq_hashes.csv
head -n 5 ./dataset_hashes/imagenette_train_pdq_hashes.csv
```

Expected columns:
- `image`
- `hash_bin`
- `hash_hex`

## 6. Exact-Match Smoke Test First

This is the recommended first execution path.

Success criterion:
- normalized target Hamming distance `<= 0.0`
- meaning exact 256-bit target hash match

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_collision_whitebox_surrogate.py \
  --source ./inputs/pdq_subset_10 \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --sample_limit 2 \
  --max_steps 50 \
  --threads 1 \
  --target_hamming 0.0 \
  --step_log_csv ./logs/attack_steps_pdq_collision_exact_smoketest.csv \
  --output_folder ./collision_attack_outputs_pdq_exact_smoketest \
  --attack_type collision_whitebox_pdq_exact_smoketest
```

Quick check:

```bash
wc -l ./logs/attack_steps_pdq_collision_exact_smoketest.csv
tail -n 20 ./logs/attack_steps_pdq_collision_exact_smoketest.csv
```

What to look for:
- the terminal should end with:
  `[pdq-collision] Finished. Logged ... data rows to ...`
- `target_dist_raw` decreasing
- `target_dist_norm` decreasing
- `success` remains `0` unless an exact collision is found

If you see only the header and `__HYPERPARAMS__` row:
- the run was interrupted too early, or
- the script printed an exception during image processing

The current script now also:
- prints per-image exceptions to stderr
- warns if you are appending to an existing log file

For a clean smoke test, remove the old log first:

```bash
rm -f ./logs/attack_steps_pdq_collision_exact_smoketest.csv
rm -rf ./collision_attack_outputs_pdq_exact_smoketest
```

## 7. Exact-Match Main Subset Run

If the smoke test behaves correctly, run the full 10-image pilot:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_pdq_collision_subset.py \
  --source ./inputs/pdq_subset_10 \
  --target_source ./data/imagenette2-320/train \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --sample_limit 10 \
  --learning_rate 0.002 \
  --ssim_weight 5.0 \
  --temperature 20.0 \
  --max_steps 1000 \
  --target_hamming 0.0 \
  --threads 2
```

Outputs:
- `./logs/attack_steps_pdq_collision_subset.csv`
- `./collision_attack_outputs_pdq_subset/`

Inspect:

```bash
wc -l ./logs/attack_steps_pdq_collision_subset.csv
tail -n 20 ./logs/attack_steps_pdq_collision_subset.csv
```

Expected terminal ending:

```text
[pdq-collision] Finished. Logged ... data rows to ./logs/attack_steps_pdq_collision_subset.csv
```

Basic success count:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('./logs/attack_steps_pdq_collision_subset.csv')
df = df[df['image_id'] != '__HYPERPARAMS__']
print(df.groupby('image_id')['success'].max().value_counts(dropna=False))
PY
```

## 8. If Exact Match Fails, Try Threshold Match

This is the fallback path.

PDQ is often used with a matching threshold, so this is still meaningful even if exact 256-bit collision is too hard.

Recommended thresholds to try in order:
- `0.08`
- `0.10`
- `0.12`
- `0.30`

### 8A. Threshold-Match Smoke Test At `0.08`

Clean old smoke-test outputs first:

```bash
rm -f ./logs/attack_steps_pdq_collision_threshold_smoketest.csv
rm -rf ./collision_attack_outputs_pdq_threshold_smoketest
```

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_collision_whitebox_surrogate.py \
  --source ./inputs/pdq_subset_10 \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --sample_limit 2 \
  --max_steps 100 \
  --threads 1 \
  --target_hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_collision_threshold_smoketest.csv \
  --output_folder ./collision_attack_outputs_pdq_threshold_smoketest \
  --attack_type collision_whitebox_pdq_threshold_smoketest
```

### 8B. Threshold-Match Main Subset Run At `0.08`

If you want a fresh subset log instead of append behavior:

```bash
rm -f ./logs/attack_steps_pdq_collision_subset.csv
rm -rf ./collision_attack_outputs_pdq_subset
```

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_pdq_collision_subset.py \
  --source ./inputs/pdq_subset_10 \
  --target_source ./data/imagenette2-320/train \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --sample_limit 10 \
  --learning_rate 0.002 \
  --ssim_weight 5.0 \
  --temperature 20.0 \
  --max_steps 1000 \
  --target_hamming 0.08 \
  --threads 2
```

### 8C. Retry At Larger Match Thresholds If Needed

At `0.10`:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_pdq_collision_subset.py \
  --source ./inputs/pdq_subset_10 \
  --target_source ./data/imagenette2-320/train \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --sample_limit 10 \
  --max_steps 1000 \
  --target_hamming 0.10 \
  --threads 2
```

At `0.12`:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_pdq_collision_subset.py \
  --source ./inputs/pdq_subset_10 \
  --target_source ./data/imagenette2-320/train \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --sample_limit 10 \
  --max_steps 1000 \
  --target_hamming 0.12 \
  --threads 2
```

At `0.30`:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/run_pdq_collision_subset.py \
  --source ./inputs/pdq_subset_10 \
  --target_source ./data/imagenette2-320/train \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --sample_limit 10 \
  --max_steps 1000 \
  --target_hamming 0.30 \
  --threads 2
```

## 9. Direct Single-Script Runs

Use this if you want full manual control instead of the subset runner.

Exact target match:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_collision_whitebox_surrogate.py \
  --source ./inputs/pdq_subset_10 \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --learning_rate 0.002 \
  --optimizer Adam \
  --ssim_weight 5.0 \
  --temperature 20.0 \
  --max_steps 1000 \
  --sample_limit 10 \
  --threads 2 \
  --check_interval 1 \
  --target_hamming 0.0 \
  --step_log_csv ./logs/attack_steps_pdq_collision_exact_manual.csv \
  --output_folder ./collision_attack_outputs_pdq_exact_manual \
  --attack_type collision_whitebox_pdq_exact_manual
```

Threshold target match at `0.08`:

```bash
PYTHONPATH=. python fdeph_eval/attacks/pdq/pdq_collision_whitebox_surrogate.py \
  --source ./inputs/pdq_subset_10 \
  --target_hashset ./dataset_hashes/imagenette_train_pdq_hashes.csv \
  --learning_rate 0.002 \
  --optimizer Adam \
  --ssim_weight 5.0 \
  --temperature 20.0 \
  --max_steps 1000 \
  --sample_limit 10 \
  --threads 2 \
  --check_interval 1 \
  --target_hamming 0.08 \
  --step_log_csv ./logs/attack_steps_pdq_collision_T008_manual.csv \
  --output_folder ./collision_attack_outputs_pdq_T008_manual \
  --attack_type collision_whitebox_pdq_T008_manual
```

## 10. Recommended Interpretation Order

Use this order when evaluating whether PDQ collision is viable:

1. Run exact-match smoke test.
2. Run exact-match 10-image subset pilot.
3. If exact collisions are absent, inspect whether `target_dist_norm` is still trending down.
4. Retry with threshold matching at `0.08`.
5. If needed, increase to `0.10`, then `0.12`, then `0.30`.

## 11. Useful Quick Analysis Commands

Final best distance per image:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('./logs/attack_steps_pdq_collision_subset.csv')
df = df[df['image_id'] != '__HYPERPARAMS__']
best = df.groupby('image_id')['best_target_dist_norm'].min().sort_values()
print(best)
PY
```

Images that succeeded:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('./logs/attack_steps_pdq_collision_subset.csv')
df = df[df['image_id'] != '__HYPERPARAMS__']
success = df.groupby('image_id')['success'].max()
print(success[success == 1])
PY
```

Last recorded step per image:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('./logs/attack_steps_pdq_collision_subset.csv')
df = df[df['image_id'] != '__HYPERPARAMS__']
last = df.sort_values('step').groupby('image_id').tail(1)
print(last[['image_id', 'target_image_id', 'target_dist_raw', 'target_dist_norm', 'l2', 'linf', 'ssim', 'success']])
PY
```

## 12. Notes

- Exact PDQ target collisions may be significantly harder than thresholded target matches.
- The current target-selection logic chooses the nearest distinct target hash from the target hashset CSV.
- The current collision attack is white-box only.
- No existing evasion files were modified for this workflow.
- The collision script is append-only with respect to the CSV logger, so reuse of the same `--step_log_csv` path will add another `__HYPERPARAMS__` row and more step rows unless you delete the file first.
