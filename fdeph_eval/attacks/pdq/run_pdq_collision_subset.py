"""
Small runner for the isolated PDQ white-box collision workflow.

Typical flow:
1. Build a target hash CSV for a target folder.
2. Run the white-box collision attack on a source subset against that hash CSV.
"""

import argparse
import os
import pathlib
import subprocess
import sys


def threshold_tag(target_hamming: float) -> str:
    if abs(target_hamming) < 1e-12:
        return "exact"
    return f"T{target_hamming:.2f}".replace(".", "")


def main():
    parser = argparse.ArgumentParser(description="Run PDQ collision attack on a subset.")
    parser.add_argument("--source", type=str, default="./inputs/pdq_subset_10")
    parser.add_argument("--target_source", type=str, default="data/imagenette2-320/train")
    parser.add_argument("--target_hashset", type=str, default="./dataset_hashes/imagenette_train_pdq_hashes.csv")
    parser.add_argument("--sample_limit", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--ssim_weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--target_hamming", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    repo_root = str(pathlib.Path(__file__).resolve().parents[3])
    hash_builder = os.path.join(repo_root, "fdeph_eval", "attacks", "pdq", "compute_dataset_hashes_pdq.py")
    whitebox = os.path.join(repo_root, "fdeph_eval", "attacks", "pdq", "pdq_collision_whitebox_surrogate.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    tag = threshold_tag(args.target_hamming)
    step_log_csv = f"./logs/attack_steps_pdq_collision_subset_{tag}.csv"
    output_folder = f"./collision_attack_outputs_pdq_subset_{tag}"
    attack_type = f"collision_whitebox_pdq_subset_{tag}"

    if not os.path.exists(args.target_hashset):
        print(f"Target hashset not found. Building: {args.target_hashset}")
        subprocess.run(
            [
                sys.executable, hash_builder,
                "--source", args.target_source,
                "--output_path", args.target_hashset,
            ],
            check=True,
            cwd=repo_root,
            env=env,
        )

    cmd = [
        sys.executable, whitebox,
        "--source", args.source,
        "--target_hashset", args.target_hashset,
        "--learning_rate", str(args.learning_rate),
        "--ssim_weight", str(args.ssim_weight),
        "--temperature", str(args.temperature),
        "--max_steps", str(args.max_steps),
        "--target_hamming", str(args.target_hamming),
        "--sample_limit", str(args.sample_limit),
        "--threads", str(args.threads),
        "--step_log_csv", step_log_csv,
        "--output_folder", output_folder,
        "--attack_type", attack_type,
    ]
    print("Running:", " ".join(cmd))
    print(f"Log file: {step_log_csv}")
    print(f"Output folder: {output_folder}")
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
