"""
Differentiable PDQ implementation for adversarial evasion attacks.
Author: Avijit Roy — FDEPH Project

PDQ pipeline (Meta/Facebook ThreatExchange):
  1. RGB → Luminance (BT.601: 0.299R + 0.587G + 0.114B)
  2. Jarosz filter (2-pass separable box blur) → 64×64
  3. 2D DCT via matrix multiply: C @ img @ C^T  (C = 64×64 DCT-II)
  4. Extract rows 1–16, cols 1–16 = 256 low-frequency coefficients
     (row 0 = DC term is excluded — PDQ is brightness-invariant)
  5. Binarize: bit = 1 if coeff > median else 0

Surrogate for attack:
  - Steps 1–4 are differentiable
  - Step 5 replaced with sigmoid((coeff - median_orig) * temperature)
  - median_orig frozen from the ORIGINAL unperturbed image
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


def jarosz_filter_2d(image_tensor: torch.Tensor, out_size: int = 64) -> torch.Tensor:
    """
    Differentiable Jarosz filter: 2-pass separable box blur → out_size×out_size.

    Args:
        image_tensor: [1, 1, H, W] float tensor on any device.
        out_size:     target spatial size (default 64).

    Returns:
        [1, 1, out_size, out_size] tensor on the same device as input.
    """
    _, _, H, W = image_tensor.shape

    # Window sizes per PDQ spec
    window_h = max(1, (W + out_size - 1) // out_size)
    window_v = max(1, (H + out_size - 1) // out_size)

    # --- Horizontal pass: average pool along width ---
    # Reshape to [1, H, W] for avg_pool1d (treats H as channels)
    x = image_tensor.squeeze(0).squeeze(0)   # [H, W]
    x = x.unsqueeze(0)                        # [1, H, W]  — 1 "batch", H channels
    # avg_pool1d expects [N, C, L]; here N=1, C=H, L=W
    x = F.avg_pool1d(x, kernel_size=window_h, stride=window_h, padding=0,
                     ceil_mode=True)           # [1, H, W']
    x = x.unsqueeze(0)                        # [1, 1, H, W']

    # --- Vertical pass: average pool along height ---
    # Transpose so height is the last dim
    x = x.squeeze(0).permute(0, 2, 1)        # [1, W', H]
    x = F.avg_pool1d(x, kernel_size=window_v, stride=window_v, padding=0,
                     ceil_mode=True)           # [1, W', H']
    x = x.permute(0, 2, 1).unsqueeze(0)      # [1, 1, H', W']

    # --- Final resize to exactly out_size × out_size ---
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear",
                      align_corners=False)    # [1, 1, 64, 64]
    return x


def _bits_to_hex(hard_hash: torch.Tensor) -> str:
    """Convert a 256-bit binary tensor (0s and 1s) to a 64-char hex string."""
    bits = hard_hash.detach().cpu().numpy().astype(int)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:064x}"


def compute_pdq_soft(
    image_tensor: torch.Tensor,
    median_ref: Optional[float] = None,
    temperature: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Differentiable PDQ surrogate for use in gradient-based attacks.

    Args:
        image_tensor: shape [1, C, H, W], values in [-1, 1].
        median_ref:   Frozen median from the ORIGINAL image. If None, the
                      median is computed from the current image (used for the
                      hard hash / first-call baseline).
        temperature:  Sigmoid sharpness parameter.

    Returns:
        soft_hash  (torch.Tensor, shape [256]): differentiable bit approximation
                   via sigmoid; values in (0, 1).
        hard_hash  (torch.Tensor, shape [256]): hard 0/1 bits (no grad).
        median_val (float): median of the 256 DCT coefficients. If median_ref
                   was supplied this equals median_ref; otherwise it is freshly
                   computed and should be saved for subsequent calls.
    """
    device = image_tensor.device

    # ---- 1. Luminance conversion (BT.601) in [0, 1] ------------------------
    img_01 = (image_tensor + 1.0) / 2.0          # [-1,1] → [0,1], [1,C,H,W]
    if img_01.shape[1] == 3:
        weights = torch.tensor(
            [0.299, 0.587, 0.114], dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        lum = (img_01 * weights).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    else:
        lum = img_01[:, :1, :, :]

    # ---- 2. Jarosz filter → 64×64 ------------------------------------------
    img_64 = jarosz_filter_2d(lum, out_size=64)   # [1, 1, 64, 64]

    # ---- 3. 2D DCT via matrix multiply: C @ img @ C^T ----------------------
    C = _get_dct_matrix_on_device(64, device)      # [64, 64]
    img_sq = img_64.squeeze(0).squeeze(0)          # [64, 64]
    dct_result = C @ img_sq @ C.t()               # [64, 64]

    # ---- 4. Extract rows 1–16, cols 1–16 = 256 coefficients ----------------
    #        (row 0 is DC / brightness term — excluded per PDQ spec)
    coeffs = dct_result[1:17, 1:17].reshape(256)  # [256]

    # ---- 5a. Determine median threshold ------------------------------------
    if median_ref is None:
        with torch.no_grad():
            median_val = float(torch.median(coeffs).item())
    else:
        median_val = float(median_ref)

    median_tensor = torch.tensor(median_val, dtype=torch.float32, device=device)

    # ---- 5b. Soft (differentiable) bits ------------------------------------
    soft_hash = torch.sigmoid((coeffs - median_tensor) * temperature)  # [256]

    # ---- 5c. Hard bits (no gradient) ----------------------------------------
    hard_hash = (coeffs.detach() >= median_val).float()                 # [256]

    return soft_hash, hard_hash, median_val


def compute_pdq_hard(
    image_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, str, float]:
    """
    Compute the standard (non-differentiable) PDQ hash of an image.

    Args:
        image_tensor: shape [1, C, H, W], values in [-1, 1].

    Returns:
        hard_hash  (torch.Tensor, shape [256]): 0/1 bit tensor.
        hash_hex   (str):  64-character hex string (for change detection).
        median_val (float): median of the 256 DCT coefficients; pass as
                            median_ref to compute_pdq_soft during the attack.
    """
    with torch.no_grad():
        _, hard_hash, median_val = compute_pdq_soft(image_tensor, median_ref=None)
    hash_hex = _bits_to_hex(hard_hash)
    return hard_hash, hash_hex, float(median_val)
