# experiments/make_inputs_sample.py
"""
Author: Avijit Roy
Purpose:
- Deterministically sample N images from a source directory tree
- Copy them into a flat output folder (e.g., ./inputs_500)
- Ensures repeatable experiment inputs across runs and machines

Usage:
  PYTHONPATH=. python experiments/make_inputs_sample.py \
    --src ./data/imagenette2-320/val \
    --out ./inputs_500 \
    --n 500 \
    --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def stable_sort_key(p: Path) -> str:
    # stable across OS: sort by relative path string
    return str(p.as_posix())


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source directory containing images (can be nested)")
    ap.add_argument("--out", required=True, help="Output directory for flat sampled images")
    ap.add_argument("--n", type=int, required=True, help="Number of images to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--clear", action="store_true", help="Clear output directory before copying")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    if not src.exists() or not src.is_dir():
        raise RuntimeError(f"--src must be an existing directory. Got: {src}")

    imgs = list_images(src)
    imgs.sort(key=stable_sort_key)

    if len(imgs) == 0:
        raise RuntimeError(f"No images found under: {src}")

    n = min(args.n, len(imgs))

    rnd = random.Random(args.seed)
    idxs = list(range(len(imgs)))
    rnd.shuffle(idxs)
    chosen = [imgs[i] for i in idxs[:n]]

    out.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for p in out.iterdir():
            if p.is_file():
                p.unlink()

    # Copy to flat folder with collision-safe names
    for i, p in enumerate(chosen, start=1):
        rel = p.relative_to(src)
        # Include a short hash of relative path to avoid collisions
        dst_name = f"{rel.stem}__{short_hash(str(rel))}{p.suffix.lower()}"
        dst_path = out / dst_name
        shutil.copy2(p, dst_path)

    print(f"Source: {src}")
    print(f"Found: {len(imgs)} images")
    print(f"Sampled: {n} images (seed={args.seed})")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()