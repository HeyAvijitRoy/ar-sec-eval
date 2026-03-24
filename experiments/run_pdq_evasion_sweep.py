# experiments/run_pdq_evasion_sweep.py
# Author: Avijit Roy
#
# Purpose:
# - Run PDQ evasion attack (SPSA) at multiple Hamming-distance thresholds
# - Produces one long-format CSV per threshold for comparative analysis
# - Each threshold gets its own output folder to avoid cross-contamination
# - Thresholds run SEQUENTIALLY to avoid CPU resource contention
#   (SPSA is CPU-bound per image; default threads=4)
#
# Usage:
#   PYTHONPATH=. python experiments/run_pdq_evasion_sweep.py [--threads N]
#
# Outputs (in logs/):
#   attack_steps_pdq_evasion_mt500_T0.08.csv
#   attack_steps_pdq_evasion_mt500_T0.10.csv
#   attack_steps_pdq_evasion_mt500_T0.12.csv
#   attack_steps_pdq_evasion_mt500_T0.30.csv

import argparse
import datetime
import os
import subprocess
import sys

THRESHOLDS = [0.08, 0.10, 0.12, 0.30]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ATTACK_SCRIPT = os.path.join(
    REPO_ROOT, "fdeph_eval", "attacks", "pdq_evasion_steps.py"
)
INPUT_DIR = os.path.join(REPO_ROOT, "inputs", "inputs_500")
LOGS_DIR  = os.path.join(REPO_ROOT, "logs")

LR      = "0.005"
C_SPSA  = "0.01"
SSIM_W  = "5"
CHECK_INT = "1"


def output_folder_for(thresh: float) -> str:
    t_str = f"{thresh:.2f}".replace(".", "")  # e.g. 0.08 → "008"
    # Use the full decimal string as the folder suffix
    t_label = f"{thresh:.2f}"                 # e.g. "0.08"
    return os.path.join(REPO_ROOT, f"evasion_attack_outputs_pdq_T{t_label}")


def run_one(thresh: float, num_threads: int) -> bool:
    t_str    = f"{thresh:.2f}"
    log_path = os.path.join(
        LOGS_DIR, f"attack_steps_pdq_evasion_mt500_T{t_str}.csv"
    )
    out_folder = output_folder_for(thresh)

    # Fresh log file for each run
    if os.path.exists(log_path):
        os.remove(log_path)

    os.makedirs(out_folder, exist_ok=True)

    cmd = [
        sys.executable, ATTACK_SCRIPT,
        "--source",          INPUT_DIR,
        "--output_folder",   out_folder,
        "--experiment_name", f"pdq_evasion_mt500_T{t_str}",
        "--threads",         str(num_threads),
        "--check_interval",  CHECK_INT,
        "--lr",              LR,
        "--c_spsa",          C_SPSA,
        "--ssim_weight",     SSIM_W,
        "--hamming",         t_str,
        "--step_log_csv",    log_path,
        "--hash_method",     "pdq",
        "--attack_type",     "evasion",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT

    t_start = datetime.datetime.now()
    print(f"\n{'='*60}")
    print(f"  Threshold : {t_str}  |  threads : {num_threads}")
    print(f"  Output    : {out_folder}")
    print(f"  Log       : {log_path}")
    print(f"  Started   : {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env)

    t_end   = datetime.datetime.now()
    elapsed = t_end - t_start
    ok      = (result.returncode == 0)

    print(f"\n  Finished  : {t_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Elapsed   : {str(elapsed).split('.')[0]}")
    print(f"  Status    : {'OK' if ok else f'FAILED (rc={result.returncode})'}")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Sweep PDQ evasion attack (SPSA) across Hamming thresholds."
    )
    parser.add_argument(
        "--threads", dest="num_threads", type=int, default=4,
        help="Number of parallel worker threads per run (default: 4)",
    )
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)

    results = {}
    for thresh in THRESHOLDS:
        ok = run_one(thresh, args.num_threads)
        results[thresh] = ok

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SWEEP SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for thresh, ok in results.items():
        t_str    = f"{thresh:.2f}"
        log_name = f"attack_steps_pdq_evasion_mt500_T{t_str}.csv"
        out_name = f"evasion_attack_outputs_pdq_T{t_str}/"
        status   = "COMPLETED" if ok else "FAILED"
        print(f"  T={t_str}  {status:<12}  {log_name}  →  {out_name}")
        if not ok:
            all_ok = False

    print(f"{'='*60}")
    if all_ok:
        print("  All threshold runs completed successfully.")
    else:
        print("  WARNING: one or more runs failed — check logs above.")
    print()


if __name__ == "__main__":
    main()
