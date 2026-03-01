# fdeph_eval/experiments/run_nhash_evasion.py
"""
Author: Avijit Roy
Runner for NeuralHash evasion step-logging experiments.
"""

import subprocess
import sys

def main():
    cmd = [
        sys.executable,
        "fdeph_eval/attacks/nhash_evasion_steps.py",
        "--source", "./inputs",                 # change to your folder
        "--output_folder", "./evasion_attack_outputs",
        "--experiment_name", "nhash_evasion_steps",
        "--threads", "4",
        "--check_interval", "1",
        "--learning_rate", "1e-3",
        "--optimizer", "Adam",
        "--ssim_weight", "5",
        "--hamming", "0.10",                    # example threshold in normalized space
        "--step_log_csv", "./logs/attack_steps_nhash_evasion.csv",
        "--hash_method", "nhash",
        "--attack_type", "evasion",
    ]
    print("Running:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()