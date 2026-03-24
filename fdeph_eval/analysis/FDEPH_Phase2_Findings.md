# FDEPH Phase 2 - Detailed Findings: pHash Evasion Study

**Author:** Avijit Roy  
**Project:** FDEPH - Security Evaluation of Perceptual Image Hashing  
**PI:** Prof. Shweta Jain  
**Date:** March 2026  
**Status:** COMPLETE - Verified

---

## 1. Executive Summary

Phase 2 evaluated the adversarial robustness of pHash (perceptual hash), a classical
DCT-based image hashing algorithm, under a gradient-based evasion attack using an
identical experimental protocol to Phase 1 (NeuralHash). The study used 500 ImageNette
validation images across four normalized Hamming distance thresholds: T=0.08, 0.10,
0.12, and 0.30.

**Key result:** pHash was successfully evaded for **500/500 images (100%)** at every
threshold. pHash evasion is **2.8× faster in steps** and **4.9× faster in wall-clock
time** than NeuralHash evasion at the same T=0.10 threshold, while producing
*less* perceptual distortion.

---

## 2. Experimental Setup

### 2.1 Dataset
- **Source:** ImageNette validation split
- **Images:** 500 (identical to Phase 1 NeuralHash study)
- **Sampling:** deterministic seed=42 via `experiments/make_inputs_sample.py`
- **Subset:** `inputs/inputs_500` (same files as Phase 1)

### 2.2 Hash Algorithm
pHash is a DCT-based perceptual hash producing a 64-bit binary vector. The pipeline:

1. Convert to grayscale
2. Resize to 32×32 (area interpolation)
3. Apply 2D DCT via fixed matrix multiply: `C @ img @ C^T`
4. Extract top-left 8×8 block = 64 low-frequency coefficients
5. Binarize: `bit_i = 1 if coeff_i >= median(coefficients) else 0`

Hash length: 64 bits. Hex representation: 16 characters.

### 2.3 Attack Method - Differentiable Surrogate pHash

pHash is not natively differentiable due to the `sign(coeff - median)` binarization
step. A differentiable surrogate was constructed:

- Steps 1-4 are fully differentiable (DCT is a fixed linear transform)
- Step 5 replaced by: `sigmoid((coeff - median_orig) × T)` where `T=20.0`
- `median_orig` is computed from the original image once and frozen throughout the attack
- Loss: `L = -mean(|soft_hash - orig_hash_bin|)` - maximizes expected bit flips
- Visual regularizer: SSIM loss with weight 5, decayed as `0.99^i`
- Optimizer: Adam, learning rate 1e-3

**Justification for T=20:** Temperature 20 was selected to balance gradient signal
strength with sigmoid saturation. Empirical results confirm convergence within a median
of 13 steps, validating this choice. Preliminary tests with T=10 showed comparable
convergence behavior.

### 2.4 Success Criterion
Identical to Phase 1:
- Hard pHash of perturbed image ≠ hard pHash of original (hex mismatch)
- Normalized Hamming distance ≥ threshold T

### 2.5 Infrastructure
- **Script:** `fdeph_eval/attacks/phash_evasion_steps.py`
- **Core module:** `utils/phash_torch.py` (differentiable DCT + surrogate)
- **Sweep runner:** `experiments/run_phash_evasion_sweep.py`
- **Threads:** 8 parallel workers per run
- **Logging:** identical long-format CSV schema to NeuralHash logs
- **Compute:** ACCESS CI Linux allocation

---

## 3. Results

### 3.1 Success Rate

| Threshold | Images | Succeeded | Success Rate |
|-----------|--------|-----------|--------------|
| T=0.08    | 500    | 500       | **100%**     |
| T=0.10    | 500    | 500       | **100%**     |
| T=0.12    | 500    | 500       | **100%**     |
| T=0.30    | 500    | 500       | **100%**     |

All thresholds achieved 100% evasion. pHash provides no adversarial resistance
under gradient-based evasion attacks.

### 3.2 Attack Efficiency

| Threshold | Median Steps | P95 Steps | Max Steps | Median Time (ms) | P95 Time (ms) |
|-----------|-------------|-----------|-----------|-----------------|---------------|
| T=0.08    | 12          | 28        | 91        | 941             | 2,585         |
| T=0.10    | 13          | 42        | 132       | 1,194           | 3,726         |
| T=0.12    | 13          | 42        | 132       | 1,134           | 3,579         |
| T=0.30    | 90          | 405       | 1,934     | 7,382           | 34,056        |

**Note on T=0.10 vs T=0.12 identity:** Both thresholds produce identical step counts
across all 500 images. This is a quantization property of 64-bit DCT hashing, not a
bug. The minimum raw bits for T=0.10 is ⌈64×0.10⌉=7; for T=0.12 is ⌈64×0.12⌉=8.
The DCT surrogate gradient flips bits in even-numbered pairs, so the first achievable
jump is always 8 bits (d_norm=0.125), which satisfies both thresholds simultaneously.

### 3.3 Perceptual Distortion

| Threshold | Median L∞  | Mean L∞   | Median L2  | Median SSIM | SSIM < 0.99 | SSIM < 0.90 |
|-----------|-----------|-----------|-----------|-------------|-------------|-------------|
| T=0.08    | 0.0078    | 0.0082    | 1.038     | 0.9995      | 1.6%        | 0.0%        |
| T=0.10    | 0.0078    | 0.0118    | 1.364     | 0.9991      | 5.0%        | 0.2%        |
| T=0.12    | 0.0078    | 0.0118    | 1.364     | 0.9991      | 5.0%        | 0.2%        |
| T=0.30    | 0.0588    | 0.0741    | 7.318     | 0.9916      | 43.0%       | 2.6%        |

**Note on T=0.30 perceptual quality:** At the operational stress-test threshold T=0.30,
evasion remained 100% successful, but 42.6% of images exhibited SSIM below 0.99 and
2.6% exhibited SSIM below 0.90 (indicating visible distortion). At operational
thresholds T=0.08-0.12, attacks are virtually imperceptible (median SSIM > 0.999).

### 3.4 d_raw Distribution (quantization structure)

At success, `dist_raw` values are always even integers. This is a structural property:
- T=0.08: stops at {6, 8, 10, 12} bits - majority at 6 bits (66.4%)
- T=0.10/0.12: stops at {8, 10, 12, 14, 16} bits - majority at 8 bits (76.6%)
- T=0.30: stops at {20, 22, 24, 26} bits - majority at 20 bits

This produces the characteristic staircase pattern visible in Distance vs Steps plots,
in contrast to the smooth monotonic climb seen in NeuralHash plots.

---

## 4. NeuralHash vs pHash Comparison (T=0.10)

| Metric            | NeuralHash | pHash    | pHash Speedup |
|-------------------|-----------|---------|---------------|
| Median steps      | 37        | 13      | **2.8×**      |
| P95 steps         | 112       | 42      | **2.7×**      |
| Median time (ms)  | 5,878     | 1,194   | **4.9×**      |
| P95 time (ms)     | 19,032    | 3,726   | **5.1×**      |
| Median L∞         | ~0.04-0.08 | 0.0078 | **more stealthy** |
| Median SSIM       | >0.996    | 0.9991  | comparable    |
| Success rate      | 100%      | 100%    | equal         |

pHash is significantly easier to evade than NeuralHash under identical attack
conditions. Notably, pHash attacks achieve *lower* L∞ distortion, meaning they are
more imperceptible despite converging faster.

---

## 5. Integrity Verification

All checks performed and passed:

| Check | Result |
|-------|--------|
| Raw bit thresholds correct (dist_raw ≥ ceil(64×T)) | PASS - all thresholds |
| Normalization consistent (dist_norm = dist_raw / 64) | PASS - 0 errors |
| dist_norm ≥ threshold at every success row | PASS - 0 violations |
| Self-consistency (same image → same hash, 3 runs) | PASS - 5/5 images |
| Non-collision (distinct images → distinct hashes) | PASS |
| Hash length (64 bits, 16-char hex) | PASS |
| T0.10 = T0.12 explained (quantization, not bug) | CONFIRMED |
| SSIM in valid range at T=0.08-0.12 | PASS - median >0.999 |
| T=0.30 SSIM degradation disclosed | NOTED in results |
| Even-integer d_raw pattern explained | DOCUMENTED |

Reproducibility: all verification checks are codified in
`fdeph_eval/analysis/phash_verification.ipynb`.

---

## 6. Thesis Write-Up Notes

### Paragraph: T=0.10 vs T=0.12 identity

> "At thresholds T=0.10 and T=0.12, identical optimization step counts were observed
> across all 500 images. This is a structural property of 64-bit DCT-based hashing
> rather than an experimental artifact. The minimum raw bits required to satisfy T=0.10
> is ⌈64×0.10⌉=7, and for T=0.12 is ⌈64×0.12⌉=8. Because the DCT surrogate gradient
> drives correlated coefficient pairs across the median threshold, bit flips occur in
> even-numbered increments. The first achievable jump is always 8 bits (d_norm=0.125),
> which simultaneously satisfies both thresholds at the same optimization step. Both
> thresholds are reported for completeness and direct comparability with NeuralHash
> results, with this quantization property explicitly noted."

### Paragraph: staircase distance pattern

> "Unlike NeuralHash, which exhibits smooth monotonic convergence in the distance vs
> steps plot, pHash displays a characteristic staircase pattern. This reflects the
> discrete quantization structure of 64-bit DCT hashing: hash bits change only when
> DCT coefficients cross the frozen median threshold, and the surrogate gradient drives
> coefficient pairs simultaneously, producing jumps in even-numbered bit increments.
> The resulting plateaus between staircase steps indicate optimization pressure
> accumulating in the continuous coefficient space before the next discrete threshold
> crossing. This is a fundamental behavioral difference between neural and classical
> perceptual hashing under gradient-based attack."

### Paragraph: T=0.30 perceptual quality

> "At the stress-test threshold T=0.30 - the operational value used in prior literature
> [Struppek et al., 2022; Jain, 2025; Madden et al., 2024] - evasion remained 100%
> successful, but perceptual quality degraded measurably. 42.6% of adversarial images
> exhibited SSIM below 0.99, and 2.6% exhibited SSIM below 0.90, indicating visible
> distortion in a subset of cases. Maximum L∞ reached 0.337 for one image. This
> suggests that while pHash evasion is feasible at T=0.30, attack stealth is not
> guaranteed at this perturbation level. At operational thresholds T=0.08-0.12,
> attacks were virtually imperceptible (median SSIM > 0.999, median L∞ = 0.0078)."

### Paragraph: attack method justification

> "pHash is not natively differentiable due to the sign-based binarization of DCT
> coefficients. A differentiable surrogate was constructed by replacing the hard
> threshold with a sigmoid function: σ((c_i − m_orig) × τ), where m_orig is the
> median of the original image's 64 DCT coefficients, frozen throughout the attack,
> and τ=20 controls sigmoid sharpness. This preserves gradient flow through the
> fixed DCT transform while approximating the discrete binarization step. The median
> anchor ensures that the surrogate always targets the same reference threshold as the
> hard hash, maintaining alignment between optimization objective and success criterion.
> Temperature τ=20 was selected empirically; convergence within a median of 13 steps
> across 500 images confirms the adequacy of this choice."

---

## 7. Artifacts Produced

### CSV Logs
```
logs/attack_steps_phash_evasion_mt500_T0.08.csv
logs/attack_steps_phash_evasion_mt500_T0.10.csv
logs/attack_steps_phash_evasion_mt500_T0.12.csv
logs/attack_steps_phash_evasion_mt500_T0.30.csv
```

### Analysis Tables
```
fdeph_eval/analysis/tables/phash_summary_stats.csv
fdeph_eval/analysis/tables/phash_threshold_sweep.csv
fdeph_eval/analysis/tables/phash_time_to_success.csv
fdeph_eval/analysis/tables/nhash_vs_phash_comparison.csv
fdeph_eval/analysis/tables/nhash_vs_phash_speedup_T010.csv
fdeph_eval/analysis/tables/phash_linf_summary.csv
```

### Figures
```
fdeph_eval/analysis/figures/phash_distance_vs_steps.png
fdeph_eval/analysis/figures/phash_distance_vs_time.png
fdeph_eval/analysis/figures/phash_success_rate_vs_steps.png
fdeph_eval/analysis/figures/phash_success_rate_vs_time.png
fdeph_eval/analysis/figures/phash_time_to_success_hist.png
fdeph_eval/analysis/figures/phash_threshold_sweep_bars.png
fdeph_eval/analysis/figures/nhash_vs_phash_success_rate_vs_steps.png
fdeph_eval/analysis/figures/nhash_vs_phash_success_rate_vs_time.png
fdeph_eval/analysis/figures/phash_ssim_distributions.png
fdeph_eval/analysis/figures/phash_adversarial_visual_inspection.png
```

### Notebooks
```
fdeph_eval/analysis/phash_attack_efficiency_analysis.ipynb
fdeph_eval/analysis/phash_verification.ipynb
```

### Core Code
```
utils/phash_torch.py
fdeph_eval/attacks/phash_evasion_steps.py
experiments/hash_self_consistency_phash.py
experiments/run_phash_evasion_sweep.py
```
