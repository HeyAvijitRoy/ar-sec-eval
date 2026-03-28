"""
fdeph_eval/attacks/pdq/run_pdq_whitebox_sweep.py
Author: Avijit Roy — FDEPH Project

Full PDQ white-box surrogate evasion sweep on inputs_500.
Runs T=0.08, T=0.10, T=0.12, T=0.30 sequentially.
White-box only — black-box is tabled for now.

Key differences from run_subset_attacks_pdq.py:
  - White-box only (no black-box)
  - sample_limit=500 (full inputs_500 set)
  - Separate output folder per threshold (lesson from Phase 2)
  - Increased max_steps for T=0.30 (requires ~77 bits vs ~22 for T=0.08)
  - Gates on source folder and logs dir existence
  - Prints per-threshold start/finish with elapsed time

Usage:
  PYTHONPATH=. python fdeph_eval/attacks/pdq/run_pdq_whitebox_sweep.py
  PYTHONPATH=. python fdeph_eval/attacks/pdq/run_pdq_whitebox_sweep.py --threads 8
  PYTHONPATH=. python fdeph_eval/attacks/pdq/run_pdq_whitebox_sweep.py --thresholds 0.08 0.10
"""

import argparse
import datetime
import os
import subprocess
import sys

# ---- Repo root and script paths ---------------------------------------------
REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WHITEBOX   = os.path.join(REPO_ROOT, "fdeph_eval", "attacks", "pdq",
                          "pdq_evasion_whitebox_surrogate.py")

# ---- Source images ----------------------------------------------------------
INPUT_DIR  = os.path.join(REPO_ROOT, "inputs", "inputs_500")

# ---- Thresholds and step budgets --------------------------------------------
# T=0.08/0.10/0.12 all converged in ~11-30 steps on pilot.
# T=0.30 needs ~77 bits flipped vs ~22 for T=0.08 — give it more room.
THRESHOLD_CONFIG = {
    0.08: {"max_steps": 500,  "label": "T=0.08"},
    0.10: {"max_steps": 500,  "label": "T=0.10"},
    0.12: {"max_steps": 500,  "label": "T=0.12"},
    0.30: {"max_steps": 1000, "label": "T=0.30"},
}

# ---- Shared hyperparameters (validated on pilot) ----------------------------
LR          = "0.002"
OPTIMIZER   = "Adam"
SSIM_WEIGHT = "5.0"
TEMPERATURE = "20.0"
CHECK_INT   = "1"


def check_prerequisites(source_dir):
    """Gate on source folder existence before launching anything."""
    if not os.path.isdir(source_dir):
        print(f"\nERROR: Source folder not found: {source_dir}")
        print("Run experiments/make_inputs_sample.py first to build inputs_500.")
        return False
    n_images = len([f for f in os.listdir(source_dir)
                    if os.path.isfile(os.path.join(source_dir, f))])
    if n_images == 0:
        print(f"\nERROR: No images found in {source_dir}")
        return False
    print(f"Source: {source_dir} ({n_images} images found)")
    return True


def run_threshold(threshold, num_threads, source_dir):
    """Run white-box attack for a single threshold. Returns True on success."""
    cfg    = THRESHOLD_CONFIG[threshold]
    t_str  = f"{threshold:.2f}"
    label  = cfg["label"]

    log_path   = os.path.join(REPO_ROOT, "logs",
                              f"attack_steps_pdq_whitebox_mt500_T{t_str}.csv")
    out_folder = os.path.join(REPO_ROOT,
                              f"evasion_attack_outputs_pdq_whitebox_T{t_str}")

    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    cmd = [
        sys.executable, WHITEBOX,
        "--source",        source_dir,
        "--sample_limit",  "500",          # CRITICAL: default is 10
        "--hamming",       t_str,
        "--max_steps",     str(cfg["max_steps"]),
        "--learning_rate", LR,
        "--optimizer",     OPTIMIZER,
        "--ssim_weight",   SSIM_WEIGHT,
        "--temperature",   TEMPERATURE,
        "--check_interval", CHECK_INT,
        "--threads",       str(num_threads),
        "--step_log_csv",  log_path,
        "--output_folder", out_folder,
        "--hash_method",   "pdq",
        "--attack_type",   f"evasion_whitebox_pdq_{t_str}",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT

    t_start = datetime.datetime.now()
    print(f"\n{'='*60}")
    print(f"  PDQ White-Box Evasion — {label}")
    print(f"  max_steps : {cfg['max_steps']}")
    print(f"  threads   : {num_threads}")
    print(f"  log       : logs/attack_steps_pdq_whitebox_mt500_T{t_str}.csv")
    print(f"  output    : evasion_attack_outputs_pdq_whitebox_T{t_str}/")
    print(f"  started   : {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env)

    t_end   = datetime.datetime.now()
    elapsed = t_end - t_start
    ok      = (result.returncode == 0)

    print(f"\n  finished  : {t_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  elapsed   : {str(elapsed).split('.')[0]}")
    print(f"  status    : {'OK' if ok else f'FAILED (rc={result.returncode})'}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="PDQ white-box evasion sweep — inputs_500, T=0.08/0.10/0.12/0.30")
    ap.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.08, 0.10, 0.12, 0.30],
        help="Thresholds to sweep (default: 0.08 0.10 0.12 0.30)")
    ap.add_argument(
        "--threads", type=int, default=8,
        help="Parallel worker threads per threshold run (default: 8)")
    ap.add_argument(
        "--source", type=str, default=INPUT_DIR,
        help=f"Source image folder (default: {INPUT_DIR})")
    args = ap.parse_args()

    # Validate thresholds
    for t in args.thresholds:
        if t not in THRESHOLD_CONFIG:
            print(f"ERROR: Unknown threshold {t}. Choose from {list(THRESHOLD_CONFIG.keys())}")
            sys.exit(1)

    # Gate on source folder
    if not check_prerequisites(args.source):
        sys.exit(1)

    print(f"\nRunning PDQ white-box sweep: {args.thresholds}")
    print(f"Threads per run : {args.threads}")
    print(f"Source          : {args.source}")

    results = {}
    for t in args.thresholds:
        ok = run_threshold(t, args.threads, args.source)
        results[t] = ok

    # ---- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  PDQ WHITE-BOX SWEEP SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for t, ok in results.items():
        t_str  = f"{t:.2f}"
        status = "COMPLETED" if ok else "FAILED"
        log    = f"logs/attack_steps_pdq_whitebox_mt500_T{t_str}.csv"
        print(f"  T={t_str}  {status:<12}  {log}")
        if not ok:
            all_ok = False

    print(f"{'='*60}")
    if all_ok:
        print("  All thresholds completed successfully.")
    else:
        print("  WARNING: one or more thresholds failed — check output above.")
    print()


if __name__ == "__main__":
    main()
