# experiments/hash_self_consistency_phash.py
# Author: Avijit Roy
#
# Purpose:
# - Verify pHash determinism: same image should always produce the same hash
# - Basic non-collision check: two distinct images should produce different hashes
# - Validates that hash length is 64 bits and hex is 16 characters
#
# Usage:
#   PYTHONPATH=. python experiments/hash_self_consistency_phash.py
#
# Exit codes: 0 = PASSED, 1 = FAILED

import os
import sys

import torch

from utils.phash_torch import compute_phash_hard
from utils.image_processing import load_and_preprocess_img

INPUT_DIR = "./inputs/inputs_50"
N_IMAGES = 5
N_REPEAT = 3   # number of times to hash each image


def main():
    # ---- Locate input images -----------------------------------------------
    if not os.path.isdir(INPUT_DIR):
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        print("Create it with: PYTHONPATH=. python experiments/make_inputs_sample.py")
        sys.exit(1)

    all_files = sorted(
        [
            os.path.join(INPUT_DIR, f)
            for f in os.listdir(INPUT_DIR)
            if os.path.isfile(os.path.join(INPUT_DIR, f))
        ]
    )
    if len(all_files) < 2:
        print(f"ERROR: Need at least 2 images in {INPUT_DIR}, found {len(all_files)}")
        sys.exit(1)

    images = all_files[:N_IMAGES]
    device = torch.device("cpu")

    failures = []

    # ---- Self-consistency check --------------------------------------------
    print(f"Checking pHash self-consistency on {len(images)} images "
          f"({N_REPEAT} calls each) ...\n")

    first_hex = None
    for img_path in images:
        name = os.path.basename(img_path)
        tensor = load_and_preprocess_img(img_path, device, resize=True)
        hashes = []
        for _ in range(N_REPEAT):
            hard_hash, hash_hex, _ = compute_phash_hard(tensor)
            hashes.append(hash_hex)

        if len(set(hashes)) != 1:
            failures.append(
                f"  INCONSISTENT: {name} produced {len(set(hashes))} "
                f"different hashes: {hashes}"
            )
        else:
            print(f"  OK  {name}  →  {hashes[0]}")

        if first_hex is None:
            first_hex = hashes[0]

    # ---- Hash-length check (first image) -----------------------------------
    tensor0 = load_and_preprocess_img(images[0], device, resize=True)
    hard_hash0, hex0, _ = compute_phash_hard(tensor0)

    print(f"\nHash length : {len(hard_hash0)} bits")
    print(f"Hex length  : {len(hex0)} characters")
    print(f"Example hex : {hex0}  (image: {os.path.basename(images[0])})")

    if len(hard_hash0) != 64:
        failures.append(f"  WRONG HASH LENGTH: expected 64, got {len(hard_hash0)}")
    if len(hex0) != 16:
        failures.append(f"  WRONG HEX LENGTH: expected 16, got {len(hex0)}")

    # ---- Non-collision check (image A vs image B) --------------------------
    tensor1 = load_and_preprocess_img(images[1], device, resize=True)
    _, hex1, _ = compute_phash_hard(tensor1)

    print(f"\nNon-collision check:")
    print(f"  Image A: {os.path.basename(images[0])}  →  {hex0}")
    print(f"  Image B: {os.path.basename(images[1])}  →  {hex1}")

    if hex0 == hex1:
        failures.append(
            f"  COLLISION: image A and image B have the same hash ({hex0})"
        )
    else:
        print("  OK  A ≠ B")

    # ---- Result ------------------------------------------------------------
    print()
    if failures:
        print("pHash self-consistency: FAILED")
        for msg in failures:
            print(msg)
        sys.exit(1)
    else:
        print("pHash self-consistency: PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
