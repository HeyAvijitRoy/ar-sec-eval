"""Differentiable and hard-oracle pHash-256 implementation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.fft import dctn

_PHASH256_DCT_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def build_dct_matrix_64(device: torch.device | str) -> torch.Tensor:
    """
    Return the 64x64 orthonormal DCT-II matrix on the requested device.
    """
    phash256_device = torch.device(device)
    phash256_key = (64, str(phash256_device))
    if phash256_key not in _PHASH256_DCT_CACHE:
        phash256_k = torch.arange(64, dtype=torch.float32).unsqueeze(1)
        phash256_i = torch.arange(64, dtype=torch.float32).unsqueeze(0)
        phash256_dct = math.sqrt(2.0 / 64.0) * torch.cos(
            math.pi * phash256_k * (2 * phash256_i + 1) / (2 * 64)
        )
        phash256_dct[0, :] = math.sqrt(1.0 / 64.0)
        _PHASH256_DCT_CACHE[phash256_key] = phash256_dct.to(phash256_device)
    return _PHASH256_DCT_CACHE[phash256_key]


def _phash256_bits_to_hex(phash256_bits: torch.Tensor) -> str:
    phash256_value = 0
    for phash256_bit in phash256_bits.detach().cpu().numpy().astype(int):
        phash256_value = (phash256_value << 1) | int(phash256_bit)
    return f"{phash256_value:064x}"


def _phash256_load_image_rgb(image_path: str | os.PathLike[str]) -> np.ndarray:
    with Image.open(image_path) as phash256_image:
        return np.asarray(phash256_image.convert("RGB"), dtype=np.float32) / 255.0


def _phash256_rgb_to_gray(phash256_rgb: np.ndarray) -> np.ndarray:
    phash256_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    return np.tensordot(phash256_rgb, phash256_weights, axes=([-1], [0])).astype(
        np.float32
    )


def _phash256_resize_gray_numpy(phash256_gray: np.ndarray) -> np.ndarray:
    phash256_gray_img = Image.fromarray(phash256_gray.astype(np.float32), mode="F")
    phash256_gray_64 = phash256_gray_img.resize((64, 64), resample=Image.BILINEAR)
    return np.asarray(phash256_gray_64, dtype=np.float32)


def _phash256_lower_median_numpy(phash256_values: np.ndarray) -> float:
    phash256_flat = np.asarray(phash256_values, dtype=np.float32).reshape(-1)
    phash256_index = (phash256_flat.size - 1) // 2
    phash256_partitioned = np.partition(phash256_flat, phash256_index)
    return float(phash256_partitioned[phash256_index])


def _phash256_hard_details(
    image_path: str | os.PathLike[str],
) -> Tuple[torch.Tensor, str, float, np.ndarray]:
    phash256_rgb = _phash256_load_image_rgb(image_path)
    phash256_gray = _phash256_rgb_to_gray(phash256_rgb)
    phash256_gray_64 = _phash256_resize_gray_numpy(phash256_gray)
    phash256_dct = dctn(phash256_gray_64, type=2, norm="ortho")
    phash256_coeffs = phash256_dct[:16, :16].reshape(256).astype(np.float32)
    phash256_median = _phash256_lower_median_numpy(phash256_coeffs)
    phash256_bits = torch.from_numpy(
        (phash256_coeffs >= phash256_median).astype(np.float32)
    )
    phash256_hex = _phash256_bits_to_hex(phash256_bits)
    return phash256_bits, phash256_hex, phash256_median, phash256_coeffs


def compute_phash256_hard(
    image_path: str | os.PathLike[str],
) -> Tuple[torch.Tensor, str]:
    """
    Compute the hard pHash-256 oracle using SciPy DCT on a file path.
    """
    phash256_bits, phash256_hex, _, _ = _phash256_hard_details(image_path)
    return phash256_bits, phash256_hex


def _phash256_extract_coeffs_torch(phash256_img_tensor: torch.Tensor) -> torch.Tensor:
    if phash256_img_tensor.ndim != 4 or phash256_img_tensor.shape[0] != 1:
        raise ValueError("phash256_img_tensor must have shape [1, C, H, W].")

    phash256_device = phash256_img_tensor.device
    phash256_img_01 = (phash256_img_tensor + 1.0) / 2.0

    if phash256_img_01.shape[1] == 3:
        phash256_weights = torch.tensor(
            [0.2989, 0.5870, 0.1140], dtype=torch.float32, device=phash256_device
        ).view(1, 3, 1, 1)
        phash256_gray = (phash256_img_01 * phash256_weights).sum(dim=1, keepdim=True)
    else:
        phash256_gray = phash256_img_01[:, :1, :, :]

    phash256_gray_64 = F.interpolate(
        phash256_gray, size=(64, 64), mode="bilinear", align_corners=False
    )
    phash256_dct_matrix = build_dct_matrix_64(phash256_device)
    phash256_square = phash256_gray_64.squeeze(0).squeeze(0)
    phash256_dct = phash256_dct_matrix @ phash256_square @ phash256_dct_matrix.t()
    return phash256_dct[:16, :16].reshape(256)


def _phash256_load_image_tensor(
    image_path: str | os.PathLike[str],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    phash256_rgb = _phash256_load_image_rgb(image_path)
    phash256_arr = (phash256_rgb * 2.0 - 1.0).transpose(2, 0, 1)
    phash256_tensor = torch.tensor(phash256_arr, dtype=torch.float32).unsqueeze(0)
    return phash256_tensor.to(device)


def compute_phash256_soft(
    img_tensor: torch.Tensor,
    median_ref: float,
    tau: float = 20,
) -> torch.Tensor:
    """
    Compute the differentiable pHash-256 surrogate bits in (0, 1).
    """
    phash256_coeffs = _phash256_extract_coeffs_torch(img_tensor)
    phash256_median_ref = torch.as_tensor(
        float(median_ref), dtype=torch.float32, device=img_tensor.device
    )
    return torch.sigmoid((phash256_coeffs - phash256_median_ref) * float(tau))


if __name__ == "__main__":
    phash256_repo_root = Path(__file__).resolve().parent.parent
    phash256_input_dir = phash256_repo_root / "inputs" / "inputs_500"
    phash256_candidates = sorted(
        phash256_path
        for phash256_path in phash256_input_dir.iterdir()
        if phash256_path.is_file()
    )
    if not phash256_candidates:
        raise FileNotFoundError(f"No input images found in {phash256_input_dir}")

    phash256_image_path = phash256_candidates[0]
    print(f"Smoke test image: {phash256_image_path}")

    phash256_hashes = []
    for phash256_run_idx in range(3):
        phash256_bits, phash256_hex = compute_phash256_hard(phash256_image_path)
        phash256_hashes.append(phash256_hex)
        print(
            f"Run {phash256_run_idx + 1}: "
            f"hex={phash256_hex} | bit_count={phash256_bits.numel()}"
        )

    phash256_consistent = len(set(phash256_hashes)) == 1
    print(f"Self-consistency over 3 runs: {'PASS' if phash256_consistent else 'FAIL'}")
    if not phash256_consistent:
        raise SystemExit(1)
