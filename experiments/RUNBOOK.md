# NeuralHash Attack Efficiency – Runbook

Author: Avijit Roy  
Project: FDEPH – Attack Efficiency Analysis  
Environment: ACCESS CI (Ubuntu + GPU)  

---

# 0. Environment & Authenticity Verification (One-Time)

## 0.1 Verify NeuralHash Seed Loads

```bash
PYTHONPATH=. python -c "from utils.hashing import load_hash_matrix; load_hash_matrix(); print('seed OK')"
```

Expected output:

```
seed OK
```

---

## 0.2 Verify Seed File Integrity (Freeze Evidence)

```bash
python - << 'PY'
import hashlib
p="models/coreml_model/neuralhash_128x96_seed1.dat"
b=open(p,"rb").read()
print("bytes:", len(b))
print("sha256:", hashlib.sha256(b).hexdigest())
PY
```

Record the SHA256 for reproducibility.

---

## 0.3 Verify Deterministic Hash (CPU Check)

```bash
PYTHONPATH=. python experiments/hash_self_consistency.py ./inputs/inputs_50/<image>.JPEG
```

Run multiple times.
Expected: identical hex values across runs.

---

# 1. Download Dataset (ImageNette – One-Time)

```bash
PYTHONPATH=. python -c "
from datasets.imagenette import ImageNette
ImageNette(root='./data', train=True, download=True)
ImageNette(root='./data', train=False, download=True)
print('imagenette downloaded')
"
```

Dataset location:

```
./data/imagenette2-320/
```

---

# 2. Create Reproducible Input Sets

All sampling is deterministic using `--seed 42`.

---

## 2.1 Create 50 Image Pilot Set

```bash
PYTHONPATH=. python experiments/make_inputs_sample.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/inputs_50 \
  --n 50 \
  --seed 42 \
  --clear
```

Verify:

```bash
ls ./inputs/inputs_50 | wc -l
```

Expected:

```
50
```

---

## 2.2 Create 500 Image Set

```bash
PYTHONPATH=. python experiments/make_inputs_sample.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/inputs_500 \
  --n 500 \
  --seed 42 \
  --clear
```

---

## 2.3 Create Larger Set (Example: 2000)

```bash
PYTHONPATH=. python experiments/make_inputs_sample.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/inputs_2000 \
  --n 2000 \
  --seed 42 \
  --clear
```

Notes:

* Same seed → nested subsets (500 ⊂ 1000 ⊂ 2000).
* Ensures reproducibility across machines.

---

# 3. Run NeuralHash Evasion Attack

---

## 3.1 MT50 (Validation Run)

```bash
rm -f ./logs/attack_steps_nhash_evasion_mt50.csv

PYTHONPATH=. python fdeph_eval/attacks/nhash_evasion_steps.py \
  --source ./inputs/inputs_50 \
  --output_folder ./evasion_attack_outputs \
  --experiment_name nhash_evasion_mt50 \
  --threads 4 \
  --check_interval 1 \
  --learning_rate 1e-3 \
  --optimizer Adam \
  --ssim_weight 5 \
  --hamming 0.10 \
  --step_log_csv ./logs/attack_steps_nhash_evasion_mt50.csv \
  --hash_method nhash \
  --attack_type evasion
```

---

## 3.2 MT500 (Main Run)

```bash
rm -f ./logs/attack_steps_nhash_evasion_mt500.csv

PYTHONPATH=. python fdeph_eval/attacks/nhash_evasion_steps.py \
  --source ./inputs/inputs_500 \
  --output_folder ./evasion_attack_outputs \
  --experiment_name nhash_evasion_mt500 \
  --threads 4 \
  --check_interval 1 \
  --learning_rate 1e-3 \
  --optimizer Adam \
  --ssim_weight 5 \
  --hamming 0.10 \
  --step_log_csv ./logs/attack_steps_nhash_evasion_mt500.csv \
  --hash_method nhash \
  --attack_type evasion
```

---

## 3.3 Threshold Sweep (0.08 / 0.10 / 0.12)

```bash
python experiments/run_nhash_evasion_sweep.py
```

Produces:

```
logs/attack_steps_nhash_evasion_mt500_T0.08.csv
logs/attack_steps_nhash_evasion_mt500_T0.10.csv
logs/attack_steps_nhash_evasion_mt500_T0.12.csv
```

Each run automatically executes sanity checks.

---

## 3.4 MT2000 (Large-Scale Run)

```bash
rm -f ./logs/attack_steps_nhash_evasion_mt2000.csv

PYTHONPATH=. python fdeph_eval/attacks/nhash_evasion_steps.py \
  --source ./inputs/inputs_2000 \
  --output_folder ./evasion_attack_outputs \
  --experiment_name nhash_evasion_mt2000 \
  --threads 4 \
  --check_interval 1 \
  --learning_rate 1e-3 \
  --optimizer Adam \
  --ssim_weight 5 \
  --hamming 0.10 \
  --step_log_csv ./logs/attack_steps_nhash_evasion_mt2000.csv \
  --hash_method nhash \
  --attack_type evasion
```

---

# 4. Sanity Check After Each Run

```bash
python experiments/sanity_check.py ./logs/<your_log_file>.csv
```

Expected:

* `Max success per image: 1`
* `Images failed: 0` (for successful experiments)
* `Images with non-consecutive steps: 0`
* `Images with decreasing elapsed_ms: 0`

---

# 5. Monitoring During Run

```bash
watch -n 2 "wc -l ./logs/attack_steps_nhash_evasion_mt500.csv"
```

---

# 6. Output Artifacts

Each run produces:

```
./logs/attack_steps_nhash_evasion_*.csv
```

Each row contains:

* image_id
* step
* elapsed_ms
* dist_raw (Hamming distance)
* dist_norm (normalized Hamming)
* L2
* L∞
* SSIM
* success flag

These files are used for:

* Distance vs Steps curves
* Distance vs Time curves
* Success rate curves
* Time-to-success histogram
* Median & 95th percentile statistics
* Threshold sweep comparison tables

---

# 7. Reproducibility Notes

* Always delete CSV before rerun.
* Never change sampling seed unless intentionally studying variance.
* Do not commit input image folders to Git.
* Record seed SHA256 and environment versions.
* Commit this RUNBOOK.md for full protocol traceability.
