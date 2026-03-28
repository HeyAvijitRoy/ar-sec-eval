"""
Run the isolated PDQ black-box and white-box attacks on a 10-image subset.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ATTACK_DIR = os.path.join(REPO_ROOT, "fdeph_eval", "attacks", "pdq")
BLACKBOX = os.path.join(ATTACK_DIR, "pdq_evasion_blackbox_spsa.py")
WHITEBOX = os.path.join(ATTACK_DIR, "pdq_evasion_whitebox_surrogate.py")


def run(cmd: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="./inputs/pdq_subset_10")
    ap.add_argument("--threshold", type=float, default=0.08)
    ap.add_argument("--blackbox_steps", type=int, default=300)
    ap.add_argument("--whitebox_steps", type=int, default=300)
    args = ap.parse_args()

    t_str = f"{args.threshold:.2f}"
    blackbox_log = os.path.join(REPO_ROOT, f"logs/attack_steps_pdq_blackbox_subset10_T{t_str}.csv")
    whitebox_log = os.path.join(REPO_ROOT, f"logs/attack_steps_pdq_whitebox_subset10_T{t_str}.csv")

    for path in [blackbox_log, whitebox_log]:
        if os.path.exists(path):
            os.remove(path)

    blackbox_cmd = [
        sys.executable, BLACKBOX,
        "--source", args.source,
        "--hamming", t_str,
        "--max_steps", str(args.blackbox_steps),
        "--step_log_csv", blackbox_log,
        "--output_folder", os.path.join(REPO_ROOT, f"evasion_attack_outputs_pdq_blackbox_T{t_str}_subset10"),
    ]
    whitebox_cmd = [
        sys.executable, WHITEBOX,
        "--source", args.source,
        "--hamming", t_str,
        "--max_steps", str(args.whitebox_steps),
        "--step_log_csv", whitebox_log,
        "--output_folder", os.path.join(REPO_ROOT, f"evasion_attack_outputs_pdq_whitebox_T{t_str}_subset10"),
    ]
    rc1 = run(blackbox_cmd)
    rc2 = run(whitebox_cmd)
    if rc1 != 0 or rc2 != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
