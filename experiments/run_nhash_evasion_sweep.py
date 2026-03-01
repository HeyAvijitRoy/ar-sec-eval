# experiments/run_nhash_evasion_sweep.py
# Author: Avijit Roy
#
# Purpose:
# - Run NeuralHash evasion experiments at multiple thresholds
# - Produces one CSV per threshold + runs sanity_check after each run
# Usage:
# - Ensure inputs/inputs_500 exists with 500 images
# - It will generate:
#       logs/attack_steps_nhash_evasion_mt500_T0.08.csv
#       logs/attack_steps_nhash_evasion_mt500_T0.10.csv
#       logs/attack_steps_nhash_evasion_mt500_T0.12.csv
# python experiments/run_nhash_evasion_sweep.py

import os
import subprocess
import sys

THRESHOLDS = [0.08, 0.10, 0.12]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ATTACK_SCRIPT = os.path.join(REPO_ROOT, "fdeph_eval", "attacks", "nhash_evasion_steps.py")
SANITY_SCRIPT = os.path.join(REPO_ROOT, "experiments", "sanity_check.py")

INPUT_DIR = os.path.join(REPO_ROOT, "inputs", "inputs_500")
OUT_DIR = os.path.join(REPO_ROOT, "evasion_attack_outputs")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

THREADS = 4
LR = "1e-3"
OPT = "Adam"
SSIM_W = "5"
CHECK_INT = "1"

def run_one(thresh: float):
    t_str = f"{thresh:.2f}"
    log_path = os.path.join(LOGS_DIR, f"attack_steps_nhash_evasion_mt500_T{t_str}.csv")

    # fresh log each run
    if os.path.exists(log_path):
        os.remove(log_path)

    cmd = [
        sys.executable, ATTACK_SCRIPT,
        "--source", INPUT_DIR,
        "--output_folder", OUT_DIR,
        "--experiment_name", f"nhash_evasion_mt500_T{t_str}",
        "--threads", str(THREADS),
        "--check_interval", CHECK_INT,
        "--learning_rate", LR,
        "--optimizer", OPT,
        "--ssim_weight", SSIM_W,
        "--hamming", t_str,
        "--step_log_csv", log_path,
        "--hash_method", "nhash",
        "--attack_type", "evasion"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT  # ensures imports work

    print("\n=== Running threshold:", t_str, "===")
    print("LOG:", log_path)
    subprocess.run(cmd, check=True, env=env)

    print("\n--- Sanity check ---")
    subprocess.run([sys.executable, SANITY_SCRIPT, log_path], check=True, env=env)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    for t in THRESHOLDS:
        run_one(t)

    print("\nAll sweep runs completed.")

if __name__ == "__main__":
    main()