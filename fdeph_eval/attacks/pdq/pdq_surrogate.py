"""
Differentiable PDQ-style surrogate used only for white-box optimization.

The exact success oracle remains ``pdqhash`` via ``utils.pdq_torch.compute_pdq_hard``.
This module approximates the preprocessing pipeline to obtain useful gradients:

  RGB -> luma -> box smoothing -> 64x64 resize -> 2D DCT -> 16x16 block -> median
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

_DCT_CACHE: dict = {}


def get_dct_matrix(n: int) -> torch.Tensor:
    key = (n, "cpu")
    if key not in _DCT_CACHE:
        k = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(n, dtype=torch.float32).unsqueeze(0)
        C = math.sqrt(2.0 / n) * torch.cos(math.pi * k * (2 * i + 1) / (2 * n))
        C[0, :] = math.sqrt(1.0 / n)
        _DCT_CACHE[key] = C
    return _DCT_CACHE[key]


def _get_dct_matrix_on_device(n: int, device: torch.device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _DCT_CACHE:
        _DCT_CACHE[key] = get_dct_matrix(n).to(device)
    return _DCT_CACHE[key]


def _estimate_window(old_dim: int, new_dim: int = 64) -> int:
    if old_dim <= new_dim:
        return 1
    window = max(1, int(round(old_dim / float(new_dim))))
    if window % 2 == 0:
        window += 1
    return window


def _to_luma(image_tensor: torch.Tensor) -> torch.Tensor:
    image_01 = (image_tensor + 1.0) / 2.0
    if image_01.shape[1] == 3:
        weights = torch.tensor(
            [0.299, 0.587, 0.114], dtype=torch.float32, device=image_tensor.device
        ).view(1, 3, 1, 1)
        return (image_01 * weights).sum(dim=1, keepdim=True)
    return image_01[:, :1, :, :]


def compute_pdq_surrogate_coefficients(image_tensor: torch.Tensor) -> torch.Tensor:
    gray = _to_luma(image_tensor)
    _, _, h, w = gray.shape
    kh = _estimate_window(h, 64)
    kw = _estimate_window(w, 64)
    if kh > 1 or kw > 1:
        gray = F.avg_pool2d(
            gray,
            kernel_size=(kh, kw),
            stride=1,
            padding=(kh // 2, kw // 2),
            count_include_pad=False,
        )
    gray = F.interpolate(gray, size=(64, 64), mode="area")
    img = gray.squeeze(0).squeeze(0)
    C = _get_dct_matrix_on_device(64, image_tensor.device)
    dct = C @ img @ C.t()
    return dct[:16, :16].reshape(-1).flip(0)


def compute_pdq_soft_surrogate(
    image_tensor: torch.Tensor,
    median_ref: Optional[float] = None,
    temperature: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    coeffs = compute_pdq_surrogate_coefficients(image_tensor)
    if median_ref is None:
        median_val = float(torch.median(coeffs.detach()).item())
    else:
        median_val = float(median_ref)
    median_tensor = torch.tensor(
        median_val, dtype=torch.float32, device=image_tensor.device
    )
    logits = (coeffs - median_tensor) * temperature
    soft_hash = torch.sigmoid(logits)
    hard_hash = (coeffs.detach() > median_val).float()
    return soft_hash, hard_hash, median_val
