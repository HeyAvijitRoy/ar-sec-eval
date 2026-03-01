# NeuralHash Attack Efficiency – Runbook

Author: Avijit Roy  
Project: FDEPH – Attack Efficiency Analysis  
Environment: ACCESS CI (Ubuntu + GPU) 

---

# 0. Environment Setup (One-Time)

Ensure NeuralHash is authentic:

```bash
PYTHONPATH=. python -c "from utils.hashing import load_hash_matrix; load_hash_matrix(); print('seed OK')"
```

Expected output:

```
seed OK
```

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

## 2.3 Create #### Image Set

```bash
PYTHONPATH=. python experiments/make_inputs_sample.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/inputs_#### \
  --n #### \
  --seed 42 \
  --clear
```

Note:

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

## 3.3 MT2000 (Big Run)

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

Run this immediately after each experiment:

```bash
python - << 'PY'
import pandas as pd, sys

path = sys.argv[1]
df = pd.read_csv(path)

success = df[df.success == 1]
grouped = success.groupby("image_id").size()

print("Total rows:", len(df))
print("Unique images:", df.image_id.nunique())
print("Images succeeded:", len(grouped))
print("Max success per image:", int(grouped.max()) if len(grouped) else 0)
PY ./logs/attack_steps_nhash_evasion_mt500.csv
```

Expected:

* `Max success per image: 1`
* Images succeeded ≈ number of images (unless some fail)

---

# 5. Monitoring During Run

Live monitor:

```bash
watch -n 2 "wc -l ./logs/attack_steps_nhash_evasion_mt500.csv"
```

---

# 6. Output Artifacts

For each run you produce:

* Long-format CSV:

  ```
  ./logs/attack_steps_nhash_evasion_*.csv
  ```

Each row contains:

* image_id
* step
* elapsed_ms
* dist_raw (Hamming)
* dist_norm (Hamming / 96)
* L2
* L∞
* SSIM
* success flag

These files are the foundation for plotting and statistical analysis.

---

# 7. Reproducibility Notes

* Always delete CSV before rerun.
* Never change seed unless intentionally studying variance.
* Do not commit input image folders to Git.
* Commit this RUNBOOK.md for protocol traceability.

