"""
Prepare a flat 10-image PDQ subset from raw Imagenette images.

Repo conventions indicate Imagenette lives under ``data/imagenette2-320`` and
attack/evaluation sampling should come from the held-out ``val`` split. By
default this script builds a class-balanced 10-image subset: one image from
 each of the 10 Imagenette classes.

Copied filenames are suffixed with ``_pdq`` before the extension.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_IMAGENETTE_CLASSES = 10


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def pick_balanced_imagenette(val_root: Path, seed: int) -> list[Path]:
    class_dirs = sorted([p for p in val_root.iterdir() if p.is_dir()])
    if len(class_dirs) != EXPECTED_IMAGENETTE_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGENETTE_CLASSES} Imagenette class folders under {val_root}, found {len(class_dirs)}."
        )

    rnd = random.Random(seed)
    chosen: list[Path] = []
    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            raise RuntimeError(f"No images found in class folder: {class_dir}")
        idx = rnd.randrange(len(images))
        chosen.append(images[idx])
    return chosen


def pick_first_n(src: Path, n: int) -> list[Path]:
    images = list_images(src)
    if len(images) < n:
        raise RuntimeError(f"Need at least {n} images under {src}; found {len(images)}.")
    return images[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="./data/imagenette2-320/val")
    ap.add_argument("--out", type=str, default="./inputs/pdq_subset_10")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["balanced_imagenette", "first_n"], default="balanced_imagenette")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    src = (repo_root / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    out_dir = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    if not src.exists() or not src.is_dir():
        raise RuntimeError(f"Source directory not found: {src}")

    if args.mode == "balanced_imagenette":
        chosen = pick_balanced_imagenette(src, args.seed)
    else:
        chosen = pick_first_n(src, args.n)

    if len(chosen) != args.n:
        raise RuntimeError(f"Expected {args.n} chosen images, got {len(chosen)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for existing in out_dir.iterdir():
            if existing.is_file():
                existing.unlink()

    for image_path in chosen:
        dst_name = f"{image_path.parent.name}_{image_path.stem}_pdq{image_path.suffix.lower()}"
        shutil.copy2(image_path, out_dir / dst_name)

    print(f"Source: {src}")
    print(f"Mode: {args.mode}")
    print(f"Prepared: {len(chosen)} images")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
