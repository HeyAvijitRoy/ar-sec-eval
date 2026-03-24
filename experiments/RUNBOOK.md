# FDEPH – Attack Efficiency Runbook

**Author:** Avijit Roy  
**Project:** FDEPH – Security Evaluation of Perceptual Image Hashing  
**Environment:** ACCESS CI (Ubuntu + GPU) | Local: Windows PC + Git Bash  
**Sync:** Git — push from local, pull on ACCESS CI  
**Last updated:** March 2026

---

## Phase status

| Phase | Target | Status |
|-------|--------|--------|
| 1 | NeuralHash evasion | ✅ COMPLETE |
| 2 | pHash evasion | ✅ COMPLETE |
| 3 | PDQ / SmartHash evasion | 🔲 PENDING |
| 4 | Three-way comparison | 🔲 PENDING |

---

## ⚠️ Important operational notes

**Output folder management:** Each threshold run writes adversarial images to
`--output_folder`. If multiple thresholds share the same folder, later runs
overwrite earlier images. The CSV logs are safe (named per run) but the PNGs
are not. **Always use a threshold-specific output folder for visualization runs.**
Example: `evasion_attack_outputs_phash_T010_viz/` not `evasion_attack_outputs_phash/`.

**Visualization image selection:** When programmatically selecting diverse examples
(fastest, best SSIM, etc.), always deduplicate by `image_id` before rendering.
Multiple criteria can resolve to the same image and cause duplicate rows.

**Timer placement:** `attack_start = time.perf_counter()` must be set before the
optimization loop, not inside it. Both `nhash_evasion_steps.py` and
`phash_evasion_steps.py` have this fix applied.

---

# 0. Environment and Authenticity Verification (one-time)

## 0.1 Verify NeuralHash seed loads

```bash
PYTHONPATH=. python -c "from utils.hashing import load_hash_matrix; load_hash_matrix(); print('seed OK')"
```

Expected: `seed OK`

---

## 0.2 Verify seed file integrity

```bash
python - << 'PY'
import hashlib
p = "models/coreml_model/neuralhash_128x96_seed1.dat"
b = open(p, "rb").read()
print("bytes:", len(b))
print("sha256:", hashlib.sha256(b).hexdigest())
PY
```

Expected SHA256: `312344458ca5468eced6f50163c09d88dbc9f3470891f1b078852b01c9a0fce9`  
Expected bytes: `49280`

---

## 0.3 Verify NeuralHash determinism

```bash
PYTHONPATH=. python experiments/hash_self_consistency.py ./inputs/inputs_50/<image>.JPEG
```

Run multiple times. Expected: identical hex values, `hamming: 0`.

---

## 0.4 Verify pHash determinism

```bash
PYTHONPATH=. python experiments/hash_self_consistency_phash.py
```

Expected output:
```
pHash self-consistency: PASSED
  Hash length : 64 bits
  Hex length  : 16 characters
  Non-collision check: OK  A ≠ B
```

---

# 1. Download Dataset (ImageNette – one-time)

```bash
PYTHONPATH=. python -c "
from datasets.imagenette import ImageNette
ImageNette(root='./data', train=True, download=True)
ImageNette(root='./data', train=False, download=True)
print('imagenette downloaded')
"
```

Dataset location: `./data/imagenette2-320/`

---

# 2. Create Reproducible Input Sets

All sampling is deterministic using `--seed 42`.  
Nested subset property: `inputs_50 ⊂ inputs_500 ⊂ inputs_2000`

## 2.1 Pilot set (50 images)

```bash
PYTHONPATH=. python experiments/make_inputs_sample.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/inputs_50 \
  --n 50 --seed 42 --clear

ls ./inputs/inputs_50 | wc -l  # expected: 50
```

## 2.2 Main set (500 images)

```bash
PYTHONPATH=. python experiments/make_inputs_sample.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/inputs_500 \
  --n 500 --seed 42 --clear
```

## 2.3 Large set (2000 images)

```bash
PYTHONPATH=. python experiments/make_inputs_sample.py \
  --src ./data/imagenette2-320/val \
  --out ./inputs/inputs_2000 \
  --n 2000 --seed 42 --clear
```

---

# 3. Phase 1 — NeuralHash Evasion

## 3.1 Pilot run (50 images, validation)

```bash
rm -f ./logs/attack_steps_nhash_evasion_mt50.csv

PYTHONPATH=. python fdeph_eval/attacks/nhash_evasion_steps.py \
  --source ./inputs/inputs_50 \
  --output_folder ./evasion_attack_outputs \
  --threads 4 --check_interval 1 \
  --learning_rate 1e-3 --optimizer Adam --ssim_weight 5 \
  --hamming 0.10 \
  --step_log_csv ./logs/attack_steps_nhash_evasion_mt50.csv \
  --hash_method nhash --attack_type evasion
```

## 3.2 Main run (500 images)

```bash
rm -f ./logs/attack_steps_nhash_evasion_mt500.csv

PYTHONPATH=. python fdeph_eval/attacks/nhash_evasion_steps.py \
  --source ./inputs/inputs_500 \
  --output_folder ./evasion_attack_outputs \
  --threads 4 --check_interval 1 \
  --learning_rate 1e-3 --optimizer Adam --ssim_weight 5 \
  --hamming 0.10 \
  --step_log_csv ./logs/attack_steps_nhash_evasion_mt500.csv \
  --hash_method nhash --attack_type evasion
```

## 3.3 Threshold sweep (T=0.08 / 0.10 / 0.12)

```bash
PYTHONPATH=. python experiments/run_nhash_evasion_sweep.py
```

Produces:
```
logs/attack_steps_nhash_evasion_mt500_T0.08.csv
logs/attack_steps_nhash_evasion_mt500_T0.10.csv
logs/attack_steps_nhash_evasion_mt500_T0.12.csv
```

## 3.4 Sanity check after each run

```bash
PYTHONPATH=. python experiments/sanity_check.py ./logs/<your_log>.csv
```

Expected (Phase 1 verified result):
```
Total rows: 23272
Unique images: 500
Images succeeded: 500
Max success per image: 1
Images failed: 0
```

Note: `Images with non-consecutive steps` and `decreasing elapsed_ms` may be
non-zero due to thread interleaving — this is expected for multi-threaded runs
and does not indicate data corruption. All per-image trajectories are individually
monotonic.

## 3.5 Phase 1 analysis notebooks

```bash
cd fdeph_eval/analysis
jupyter nbconvert --to notebook --execute attack_efficiency_analysis.ipynb \
  --output attack_efficiency_analysis_executed.ipynb
```

Produces figures in `fdeph_eval/analysis/figures/nhash_*` and tables in
`fdeph_eval/analysis/tables/nhash_*`.

---

# 4. Phase 2 — pHash Evasion

## 4.1 Pilot run (50 images, validation)

```bash
PYTHONPATH=. python fdeph_eval/attacks/phash_evasion_steps.py \
  --source ./inputs/inputs_50 \
  --output_folder ./evasion_attack_outputs_phash_pilot \
  --hamming 0.10 --threads 8 \
  --step_log_csv ./logs/attack_steps_phash_evasion_pilot50.csv \
  --hash_method phash --attack_type evasion \
  --experiment_name phash_evasion_pilot50
```

Then sanity check:
```bash
PYTHONPATH=. python experiments/sanity_check.py \
  ./logs/attack_steps_phash_evasion_pilot50.csv
```

Expected: `Images succeeded: 50`, `Images failed: 0`

## 4.2 Threshold sweep (T=0.08 / 0.10 / 0.12 / 0.30) — 500 images

```bash
PYTHONPATH=. python experiments/run_phash_evasion_sweep.py --threads 8
```

Produces:
```
logs/attack_steps_phash_evasion_mt500_T0.08.csv
logs/attack_steps_phash_evasion_mt500_T0.10.csv
logs/attack_steps_phash_evasion_mt500_T0.12.csv
logs/attack_steps_phash_evasion_mt500_T0.30.csv
```

All written to `evasion_attack_outputs_phash/` — **see output folder warning above**.

## 4.3 Targeted re-run for visualization (T=0.10 only, specific images)

When you need clean T=0.10 adversarial images for the hash visualization figure,
re-run on the selected images into a dedicated folder:

```bash
mkdir -p evasion_attack_outputs_phash_T010_viz

# Run each selected image individually
while IFS= read -r img_path; do
    PYTHONPATH=. python fdeph_eval/attacks/phash_evasion_steps.py \
        --source "$img_path" \
        --output_folder evasion_attack_outputs_phash_T010_viz \
        --hamming 0.10 --threads 1 \
        --step_log_csv ./logs/viz_phash_T010_5images.csv \
        --hash_method phash --attack_type evasion
done < /tmp/viz_images.txt
```

To generate `/tmp/viz_images.txt`:
```bash
PYTHONPATH=. python3 << 'EOF'
import csv, os, glob

ids_8bit = []
with open('logs/attack_steps_phash_evasion_mt500_T0.10.csv') as f:
    for r in csv.DictReader(f):
        if r['success'] == '1' and float(r.get('dist_raw', 0)) == 8.0:
            ids_8bit.append(r['image_id'])

with open('/tmp/viz_images.txt', 'w') as f:
    for id_ in ids_8bit[:10]:
        for ext in ['.JPEG', '.jpeg', '.jpg', '.png']:
            p = os.path.join('inputs/inputs_500', id_ + ext)
            if os.path.exists(p):
                f.write(p + '\n')
                break

print(f"Written {sum(1 for _ in open('/tmp/viz_images.txt'))} image paths")
EOF
```

## 4.4 Phase 2 analysis notebooks

```bash
cd fdeph_eval/analysis

# Main efficiency analysis
jupyter nbconvert --to notebook --execute phash_attack_efficiency_analysis.ipynb \
  --output phash_attack_efficiency_analysis_executed.ipynb

# Integrity verification (8 checks)
jupyter nbconvert --to notebook --execute phash_verification.ipynb \
  --output phash_verification_executed.ipynb

# Hash visualization (requires T010_viz folder populated)
jupyter nbconvert --to notebook --execute phash_attack_visualization_v2.ipynb \
  --output phash_attack_visualization_v2_executed.ipynb
```

Produces figures in `fdeph_eval/analysis/figures/phash_*` and tables in
`fdeph_eval/analysis/tables/phash_*`.

## 4.5 Key Phase 2 results (verified)

| Threshold | Success | Median Steps | Median Time (ms) | Median SSIM |
|-----------|---------|-------------|-----------------|-------------|
| T=0.08 | 500/500 | 12 | 941 | 0.9995 |
| T=0.10 | 500/500 | 13 | 1,194 | 0.9991 |
| T=0.12 | 500/500 | 13† | 1,134† | 0.9991 |
| T=0.30 | 500/500 | 90 | 7,382 | 0.9916 |

† T=0.10 and T=0.12 produce identical results — quantization artifact, documented.

pHash vs NeuralHash at T=0.10: **2.8× fewer steps, 4.9× faster, lower L∞ distortion**.

---

# 5. Sanity Check (general — all phases)

```bash
PYTHONPATH=. python experiments/sanity_check.py ./logs/<your_log>.csv
```

Always run after any attack. Expected for a clean run:
- `Images succeeded: N` (where N = images in source)
- `Images failed: 0`
- `Max success per image: 1`

Non-consecutive steps / decreasing elapsed_ms warnings are threading artifacts,
not data errors.

---

# 6. Monitoring During a Run

```bash
# Count success events so far
watch -n 10 'grep ",1," ./logs/<your_log>.csv | wc -l'

# Line count growth
watch -n 2 "wc -l ./logs/<your_log>.csv"
```

---

# 7. Output Artifacts Summary

## Per-run CSV schema (identical for all phases)

| Column | Description |
|--------|-------------|
| `image_id` | Image stem identifier |
| `hash_method` | `nhash` or `phash` |
| `attack_type` | `evasion` |
| `step` | Optimization step number |
| `elapsed_ms` | Wall-clock time from attack start |
| `dist_raw` | Hamming distance (raw bit count) |
| `dist_norm` | Normalized Hamming (dist_raw / hash_bits) |
| `l2` | L2 distortion in [0,1] pixel space |
| `linf` | L∞ distortion in [0,1] pixel space |
| `ssim` | SSIM between original and adversarial |
| `success` | 1 if threshold met and hash changed, else 0 |
| `source_path` | Original image path |

Row 0 of each CSV is a `__HYPERPARAMS__` row recording all run parameters.

## Analysis outputs per phase

```
fdeph_eval/analysis/figures/        ← all PNG figures
fdeph_eval/analysis/tables/         ← all CSV summary tables
fdeph_eval/analysis/notebooks/      ← executed notebooks
```

---

# 8. Reproducibility Notes

- Always delete the target CSV before re-running to avoid appending to stale data
- Never change `--seed 42` unless deliberately studying variance
- Do not commit `inputs/` folders or `data/` to Git (in `.gitignore`)
- Do commit CSV logs, figures, tables, and notebooks
- Record seed SHA256 and environment versions in `evidence/`
- Each threshold sweep should use a **dedicated output folder** for adversarial images
- This RUNBOOK is the authoritative protocol record — keep it updated

---

# 9. Phase 3 — PDQ / SmartHash (planned)

PDQ (Meta/Facebook, 256-bit DCT hash) is the proposed next target.  
SmartHash (Jain 2025, 192-bit multi-level quantization) follows once code is received.

When ready, attack scripts will follow the same pattern:
```
utils/pdq_torch.py                          ← differentiable PDQ surrogate
fdeph_eval/attacks/pdq_evasion_steps.py     ← attack script
experiments/hash_self_consistency_pdq.py    ← determinism check
experiments/run_pdq_evasion_sweep.py        ← sweep runner (T=0.08/0.10/0.12/0.30)
```

Same 500 ImageNette images. Same threshold set. Same CSV schema.  
Output folders: `evasion_attack_outputs_pdq_T{threshold}/` (one per threshold).
