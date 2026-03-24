"""
PDQ hash wrapper for FDEPH adversarial evasion attacks.
Author: Avijit Roy — FDEPH Project

PDQ (Meta/Facebook ThreatExchange) produces a 256-bit perceptual hash.
The full pipeline (Jarosz filter → 64×64 → DCT → 16×16 submatrix → binarize)
is not fully differentiable due to the Jarosz filter implementation.

This module wraps the pdqhash library's compute_float() function, which
returns the 256 raw DCT coefficients before binarization. These are used
as the oracle for SPSA-based gradient estimation in the attack.

Attack strategy: SPSA (Simultaneous Perturbation Stochastic Approximation)
  - 2 oracle calls per step regardless of image dimensions
  - Feasible: compute_float ≈ 1.3ms → ~2.6ms per step
  - Black-box attack: matches threat model in Madden et al. (2024)
"""
from typing import Tuple

import numpy as np
import torch

# Module-level cache for the pdqhash import
_pdqhash_lib = None


def _get_pdqhash():
    """Lazy-load pdqhash and cache it at module level."""
    global _pdqhash_lib
    if _pdqhash_lib is None:
        try:
            import pdqhash as _pdqhash
            _pdqhash_lib = _pdqhash
        except ImportError:
            raise ImportError(
                "pdqhash not installed. Run: pip install pdqhash --user"
            )
    return _pdqhash_lib


def bits_to_hex_256(hard_hash: torch.Tensor) -> str:
    """Convert 256-bit tensor to 64-character hex string."""
    bits = hard_hash.detach().cpu().numpy().astype(int)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:064x}"


def tensor_to_numpy_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert [1, C, H, W] tensor in [-1,1] to [H, W, 3] uint8 numpy.
    Handles grayscale (C=1) by repeating channels.
    Clamps to [0, 255] after conversion.
    """
    img = image_tensor.detach().squeeze(0)  # [C, H, W]
    img = ((img + 1.0) / 2.0 * 255.0).clamp(0, 255)
    img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def compute_pdq_float(image_np: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Get raw DCT float coefficients from pdqhash library.

    Args:
        image_np: [H, W, 3] uint8 numpy array in [0,255]
    Returns:
        float_vec: np.ndarray shape [256], raw DCT coefficients
        quality:   int quality score
    """
    _pdq = _get_pdqhash()
    return _pdq.compute_float(image_np)


def compute_pdq_hard(
    image_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, str, float]:
    """
    Compute hard PDQ hash using pdqhash library as ground truth.

    Args:
        image_tensor: [1, C, H, W] in [-1, 1]
    Returns:
        hard_hash: torch.Tensor shape [256], values 0/1
        hash_hex:  64-character hex string
        median_val: float (median of 256 DCT coefficients)
    """
    _pdq = _get_pdqhash()
    img_np = tensor_to_numpy_rgb(image_tensor)
    float_vec, quality = _pdq.compute_float(img_np)
    float_vec = np.array(float_vec, dtype=np.float32)
    median_val = float(np.median(float_vec))
    hard_hash = torch.tensor((float_vec > median_val).astype(np.float32))
    hash_hex = bits_to_hex_256(hard_hash)
    return hard_hash, hash_hex, median_val
