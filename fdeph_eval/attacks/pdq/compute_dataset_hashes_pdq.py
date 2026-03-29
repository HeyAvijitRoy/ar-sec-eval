"""
Compute PDQ hashes for every image in a file or folder tree and save as CSV.

This mirrors the existing dataset-hash utilities, but keeps the PDQ collision
workflow fully isolated under ``fdeph_eval/attacks/pdq/``.
"""

import argparse
import os
import pathlib
import sys

import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.image_processing import load_and_preprocess_img
from utils.pdq_torch import compute_pdq_hard


def iter_images(source: str):
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        images = [
            os.path.join(path, name)
            for path, _, files in os.walk(source)
            for name in files
        ]
        return sorted(images)
    raise RuntimeError(f"{source} is neither a file nor a directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute PDQ hashes for a folder of images and save to CSV."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/imagenette2-320/train",
        help="Image file or folder to compute hashes for",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help=(
            "Full path for output CSV. "
            "Default: dataset_hashes/{folder_name}_pdq_hashes.csv"
        ),
    )
    args = parser.parse_args()

    images = iter_images(args.source)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rows = []
    for img_name in tqdm(images, desc="Computing PDQ"):
        try:
            img = load_and_preprocess_img(img_name, device, resize=True)
            hard_hash, hash_hex, _ = compute_pdq_hard(img)
        except Exception:
            continue

        hash_bin = "".join("1" if b > 0.5 else "0" for b in hard_hash.tolist())
        rows.append({"image": img_name, "hash_bin": hash_bin, "hash_hex": hash_hex})

    result_df = pd.DataFrame(rows, columns=["image", "hash_bin", "hash_hex"])

    os.makedirs("./dataset_hashes", exist_ok=True)
    if args.output_path:
        out_path = args.output_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    elif os.path.isfile(args.source):
        name = pathlib.PurePath(args.source).name
        out_path = os.path.join("./dataset_hashes", f"{name}_pdq_hashes.csv")
    else:
        folder_name = pathlib.PurePath(args.source).name
        out_path = os.path.join("./dataset_hashes", f"{folder_name}_pdq_hashes.csv")

    result_df.to_csv(out_path, index=False)
    print(f"Saved {len(result_df)} hashes to {out_path}")


if __name__ == "__main__":
    main()
