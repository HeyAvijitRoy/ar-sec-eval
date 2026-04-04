"""Sanity check for pHash-256 determinism."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from phash_256.phash256_torch import compute_phash256_hard
except ModuleNotFoundError:
    from phash256_torch import compute_phash256_hard  # type: ignore


PHASH256_REPO_ROOT = Path(__file__).resolve().parent.parent
PHASH256_INPUT_DIR = PHASH256_REPO_ROOT / "inputs" / "inputs_500"
PHASH256_N_IMAGES = 5
PHASH256_N_REPEAT = 3


def main() -> None:
    if not PHASH256_INPUT_DIR.is_dir():
        print(f"ERROR: Input directory not found: {PHASH256_INPUT_DIR}")
        sys.exit(1)

    phash256_images = sorted(
        phash256_path
        for phash256_path in PHASH256_INPUT_DIR.iterdir()
        if phash256_path.is_file()
    )[:PHASH256_N_IMAGES]

    if len(phash256_images) < PHASH256_N_IMAGES:
        print(
            f"ERROR: Need at least {PHASH256_N_IMAGES} images in "
            f"{PHASH256_INPUT_DIR}, found {len(phash256_images)}"
        )
        sys.exit(1)

    phash256_failures = []
    print(
        f"Checking pHash-256 self-consistency on {len(phash256_images)} images "
        f"({PHASH256_N_REPEAT} calls each) ...\n"
    )

    for phash256_image_path in phash256_images:
        phash256_hashes = []
        for _ in range(PHASH256_N_REPEAT):
            _, phash256_hash_hex = compute_phash256_hard(phash256_image_path)
            phash256_hashes.append(phash256_hash_hex)

        if len(set(phash256_hashes)) != 1:
            phash256_failures.append(
                f"  FAIL {phash256_image_path.name} -> {phash256_hashes}"
            )
            print(f"  FAIL {phash256_image_path.name}")
        else:
            print(f"  PASS {phash256_image_path.name} -> {phash256_hashes[0]}")

    print()
    if phash256_failures:
        print("pHash-256 self-consistency: FAILED")
        for phash256_failure in phash256_failures:
            print(phash256_failure)
        sys.exit(1)

    print("pHash-256 self-consistency: PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
