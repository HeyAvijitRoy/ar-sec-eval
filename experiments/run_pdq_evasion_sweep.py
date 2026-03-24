# experiments/run_pdq_evasion_sweep.py
# Author: Avijit Roy
#
# Purpose:
# - Run PDQ evasion attack at multiple Hamming-distance thresholds
# - IMPORTANT: each threshold gets its own output folder to prevent
#   image overwriting (lesson from Phase 2)
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
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

LR = "1e-3"
OPT = "Adam"
SSIM_W = "5"
CHECK_INT = "1"
TEMPERATURE = "20.0"


def run_one(thresh: float, num_threads: int) -> bool:
    t_str = f"{thresh:.2f}"
    log_path = os.path.join(
        LOGS_DIR, f"attack_steps_pdq_evasion_mt500_T{t_str}.csv"
    )
    out_dir = os.path.join(REPO_ROOT, f"evasion_attack_outputs_pdq_T{t_str}")
    os.makedirs(out_dir, exist_ok=True)

    # Fresh log file for each run
    if os.path.exists(log_path):
        os.remove(log_path)

    cmd = [
        sys.executable, ATTACK_SCRIPT,
        "--source", INPUT_DIR,
        "--output_folder", out_dir,
        "--experiment_name", f"pdq_evasion_mt500_T{t_str}",
        "--threads", str(num_threads),
        "--check_interval", CHECK_INT,
        "--learning_rate", LR,
        "--optimizer", OPT,
        "--ssim_weight", SSIM_W,
        "--hamming", t_str,
        "--temperature", TEMPERATURE,
        "--step_log_csv", log_path,
        "--hash_method", "pdq",
        "--attack_type", "evasion",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT

    t_start = datetime.datetime.now()
    print(f"\n{'='*60}")
    print(f"  Threshold : {t_str}  |  threads : {num_threads}")
    print(f"  Log       : {log_path}")
    print(f"  Out dir   : {out_dir}")
    print(f"  Started   : {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env)

    t_end = datetime.datetime.now()
    elapsed = t_end - t_start
    ok = (result.returncode == 0)

    print(f"\n  Finished  : {t_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Elapsed   : {str(elapsed).split('.')[0]}")
    print(f"  Status    : {'OK' if ok else f'FAILED (rc={result.returncode})'}")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Sweep PDQ evasion attack across Hamming thresholds."
    )
    parser.add_argument(
        "--threads", dest="num_threads", type=int, default=8,
        help="Number of parallel worker threads per run (default: 8)",
    )
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)

    results = {}
    for thresh in THRESHOLDS:
        ok = run_one(thresh, args.num_threads)
        results[thresh] = ok

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  PDQ SWEEP SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for thresh, ok in results.items():
        t_str = f"{thresh:.2f}"
        log_name = f"attack_steps_pdq_evasion_mt500_T{t_str}.csv"
        status = "COMPLETED" if ok else "FAILED"
        print(f"  T={t_str}  {status:<12}  {log_name}")
        if not ok:
            all_ok = False

    print(f"{'='*60}")
    if all_ok:
        print("  All PDQ threshold runs completed successfully.")
    else:
        print("  WARNING: one or more PDQ runs failed — check logs above.")
    print()


if __name__ == "__main__":
    main()
