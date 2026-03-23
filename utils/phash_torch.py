"""
Differentiable pHash implementation for adversarial evasion attacks.
Author: Avijit Roy — FDEPH Project

pHash standard pipeline:
  1. Grayscale conversion
  2. Resize to 32x32 (area interpolation)
  3. 2D DCT via fixed matrix multiply
  4. Take top-left 8x8 = 64 low-freq coefficients
  5. Binarize: bit_i = 1 if coeff_i >= median(coefficients), else 0

For attack (surrogate):
  - Steps 1-4 are differentiable as written
  - Step 5 replaced with sigmoid((coeff - median_orig) * temperature)
  - median_orig is frozen from the ORIGINAL unperturbed image
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Module-level DCT matrix cache, keyed by (n, device_str)
_DCT_CACHE: dict = {}


def get_dct_matrix(n: int) -> torch.Tensor:
    """
    Returns the NxN orthonormal DCT-II matrix as a CPU float tensor.

    C[k, i] = sqrt(1/N)              for k == 0
    C[k, i] = sqrt(2/N) * cos(pi*k*(2*i+1)/(2*N))  for k > 0
    """
    key = (n, "cpu")
    if key not in _DCT_CACHE:
        k = torch.arange(n, dtype=torch.float32).unsqueeze(1)   # [n, 1]
        i = torch.arange(n, dtype=torch.float32).unsqueeze(0)   # [1, n]
        C = math.sqrt(2.0 / n) * torch.cos(math.pi * k * (2 * i + 1) / (2 * n))
        C[0, :] = math.sqrt(1.0 / n)
        _DCT_CACHE[key] = C
    return _DCT_CACHE[key]


def _get_dct_matrix_on_device(n: int, device: torch.device) -> torch.Tensor:
    """Returns the NxN DCT-II matrix on the specified device (cached)."""
    key = (n, str(device))
    if key not in _DCT_CACHE:
        _DCT_CACHE[key] = get_dct_matrix(n).to(device)
    return _DCT_CACHE[key]


def _bits_to_hex(hard_hash: torch.Tensor) -> str:
    """Convert a 64-bit binary tensor (0s and 1s) to a 16-char hex string."""
    bits = hard_hash.detach().cpu().numpy().astype(int)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def compute_phash_soft(
    image_tensor: torch.Tensor,
    median_ref: Optional[float] = None,
    temperature: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Differentiable pHash surrogate for use in gradient-based attacks.

    Args:
        image_tensor: shape [1, C, H, W], values in [-1, 1].
        median_ref:   Frozen median from the ORIGINAL image. If None, the
                      median is computed from the current image (used for the
                      hard hash / first-call baseline).
        temperature:  Sigmoid sharpness parameter.

    Returns:
        soft_hash  (torch.Tensor, shape [64]): differentiable bit approximation
                   via sigmoid; values in (0, 1).
        hard_hash  (torch.Tensor, shape [64]): hard 0/1 bits (no grad).
        median_val (float): median of the 64 DCT coefficients. If median_ref
                   was supplied this equals median_ref; otherwise it is freshly
                   computed and should be saved for subsequent calls.
    """
    device = image_tensor.device

    # ---- 1. Grayscale conversion (in [0, 1]) --------------------------------
    img_01 = (image_tensor + 1.0) / 2.0          # [-1,1] → [0,1], [1,C,H,W]
    weights = torch.tensor(
        [0.2989, 0.5870, 0.1140], dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    # Handle both RGB (C=3) and already-grey (C=1) inputs
    if img_01.shape[1] == 3:
        gray = (img_01 * weights).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    else:
        gray = img_01[:, :1, :, :]

    # ---- 2. Resize to 32×32 -------------------------------------------------
    gray_32 = F.interpolate(
        gray, size=(32, 32), mode="bilinear", align_corners=False
    )  # [1, 1, 32, 32]

    # ---- 3. 2D DCT via matrix multiply: C @ img @ C^T ----------------------
    C = _get_dct_matrix_on_device(32, device)        # [32, 32]
    img_sq = gray_32.squeeze(0).squeeze(0)           # [32, 32]
    dct_result = C @ img_sq @ C.t()                  # [32, 32]

    # ---- 4. Top-left 8×8 block → 64 coefficients ---------------------------
    coeffs = dct_result[:8, :8].reshape(64)          # [64]

    # ---- 5a. Determine median threshold ------------------------------------
    if median_ref is None:
        with torch.no_grad():
            median_val = float(torch.median(coeffs).item())
    else:
        median_val = float(median_ref)

    median_tensor = torch.tensor(median_val, dtype=torch.float32, device=device)

    # ---- 5b. Soft (differentiable) bits ------------------------------------
    soft_hash = torch.sigmoid((coeffs - median_tensor) * temperature)  # [64]

    # ---- 5c. Hard bits (no gradient) ----------------------------------------
    hard_hash = (coeffs.detach() >= median_val).float()                 # [64]

    return soft_hash, hard_hash, median_val


def compute_phash_hard(
    image_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, str, float]:
    """
    Compute the standard (non-differentiable) pHash of an image.

    Args:
        image_tensor: shape [1, C, H, W], values in [-1, 1].

    Returns:
        hard_hash  (torch.Tensor, shape [64]): 0/1 bit tensor.
        hash_hex   (str):  16-character hex string (for change detection).
        median_val (float): median of the 64 DCT coefficients; pass as
                            median_ref to compute_phash_soft during the attack.
    """
    with torch.no_grad():
        _, hard_hash, median_val = compute_phash_soft(image_tensor, median_ref=None)
    hash_hex = _bits_to_hex(hard_hash)
    return hard_hash, hash_hex, float(median_val)
