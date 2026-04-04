"""Threshold sweep runner for self-contained pHash-256 evasion."""

from __future__ import annotations

import argparse
import csv
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

try:
    from phash_256.phash256_evasion_steps import run_evasion
except ModuleNotFoundError:
    from phash256_evasion_steps import run_evasion  # type: ignore


THRESHOLDS = [0.08, 0.10, 0.12, 0.30]

PHASH256_REPO_ROOT = Path(__file__).resolve().parent.parent
PHASH256_INPUT_DIR = PHASH256_REPO_ROOT / "inputs" / "inputs_500"
PHASH256_LOGS_DIR = PHASH256_REPO_ROOT / "logs"

PHASH256_CSV_HEADER = [
    "image_id",
    "threshold",
    "success",
    "steps",
    "dist_norm",
    "dist_raw",
    "l2",
    "l_inf",
    "ssim",
    "time_ms",
    "orig_hash_hex",
    "final_hash_hex",
]


def _phash256_choose_device(phash256_device_arg: str) -> str:
    if phash256_device_arg != "auto":
        return phash256_device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _phash256_list_images(phash256_sample_limit: int) -> List[Path]:
    if not PHASH256_INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory not found: {PHASH256_INPUT_DIR}")
    phash256_images = sorted(
        phash256_path
        for phash256_path in PHASH256_INPUT_DIR.iterdir()
        if phash256_path.is_file()
    )
    return phash256_images[:phash256_sample_limit]


def _phash256_log_path(phash256_threshold: float) -> Path:
    return PHASH256_LOGS_DIR / (
        f"attack_steps_phash256_evasion_mt500_T{phash256_threshold:.2f}.csv"
    )


def run_one(
    thresh: float,
    num_threads: int,
    sample_limit: int,
    device: str,
    max_steps: int,
) -> bool:
    phash256_images = _phash256_list_images(sample_limit)
    phash256_log_path = _phash256_log_path(thresh)
    PHASH256_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if phash256_log_path.exists():
        phash256_log_path.unlink()

    phash256_start = datetime.datetime.now()
    print(f"\n{'=' * 60}")
    print(f"  Threshold : {thresh:.2f}  |  threads : {num_threads}")
    print(f"  Device    : {device}")
    print(f"  Images    : {len(phash256_images)}")
    print(f"  Log       : {phash256_log_path}")
    print(f"  Started   : {phash256_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    phash256_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as phash256_executor:
        phash256_future_map = {
            phash256_executor.submit(
                run_evasion,
                image_path=str(phash256_image_path),
                threshold=thresh,
                max_steps=max_steps,
                device=device,
            ): (phash256_index, phash256_image_path)
            for phash256_index, phash256_image_path in enumerate(phash256_images, start=1)
        }

        for phash256_future in as_completed(phash256_future_map):
            phash256_index, _ = phash256_future_map[phash256_future]
            phash256_result = phash256_future.result()
            phash256_results.append((phash256_index, phash256_result))
            phash256_status = "success" if phash256_result["success"] else "fail"
            print(
                f"T={thresh:.2f} | image {phash256_index}/{len(phash256_images)} | "
                f"{phash256_status} | {phash256_result['steps']} | "
                f"{phash256_result['dist_norm']:.6f}"
            )

    phash256_results.sort(key=lambda phash256_item: phash256_item[0])
    with phash256_log_path.open("w", newline="", encoding="utf-8") as phash256_csv_file:
        phash256_writer = csv.DictWriter(
            phash256_csv_file,
            fieldnames=PHASH256_CSV_HEADER,
        )
        phash256_writer.writeheader()
        for _, phash256_result in phash256_results:
            phash256_writer.writerow(phash256_result)

    phash256_end = datetime.datetime.now()
    phash256_elapsed = phash256_end - phash256_start
    print(f"\n  Finished  : {phash256_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Elapsed   : {str(phash256_elapsed).split('.')[0]}")
    print("  Status    : OK")
    return True


def main() -> None:
    phash256_parser = argparse.ArgumentParser(
        description="Sweep pHash-256 evasion attack across Hamming thresholds."
    )
    phash256_parser.add_argument(
        "--threads",
        dest="num_threads",
        type=int,
        default=8,
        help="Number of parallel worker threads per run (default: 8)",
    )
    phash256_parser.add_argument(
        "--sample_limit",
        type=int,
        default=500,
        help="Maximum number of images to process from inputs_500 (default: 500)",
    )
    phash256_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for attacks: auto, cpu, cuda:0, ...",
    )
    phash256_parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Maximum optimization steps per image (default: 2000)",
    )
    phash256_parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=THRESHOLDS,
        help="Thresholds to sweep (default: 0.08 0.10 0.12 0.30)",
    )
    phash256_args = phash256_parser.parse_args()

    phash256_device = _phash256_choose_device(phash256_args.device)
    phash256_results = {}
    for phash256_thresh in phash256_args.thresholds:
        phash256_ok = run_one(
            thresh=phash256_thresh,
            num_threads=phash256_args.num_threads,
            sample_limit=phash256_args.sample_limit,
            device=phash256_device,
            max_steps=phash256_args.max_steps,
        )
        phash256_results[phash256_thresh] = phash256_ok

    print(f"\n{'=' * 60}")
    print("  SWEEP SUMMARY")
    print(f"{'=' * 60}")
    for phash256_thresh, phash256_ok in phash256_results.items():
        phash256_status = "COMPLETED" if phash256_ok else "FAILED"
        print(
            f"  T={phash256_thresh:.2f}  {phash256_status:<12}  "
            f"{_phash256_log_path(phash256_thresh).name}"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
