"""Self-contained per-image evasion attack for pHash-256."""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from phash_256.phash256_torch import (
        _phash256_hard_details,
        compute_phash256_hard,
        compute_phash256_soft,
    )
except ModuleNotFoundError:
    from phash256_torch import (  # type: ignore
        _phash256_hard_details,
        compute_phash256_hard,
        compute_phash256_soft,
    )


def _phash256_load_and_preprocess_img(
    phash256_img_path: str | os.PathLike[str],
    phash256_device: torch.device | str,
    phash256_resize: bool = True,
) -> torch.Tensor:
    with Image.open(phash256_img_path) as phash256_image:
        phash256_image = phash256_image.convert("RGB")
        if phash256_resize:
            phash256_image = phash256_image.resize((360, 360), resample=Image.BILINEAR)
        phash256_arr = np.asarray(phash256_image, dtype=np.float32) / 255.0

    phash256_arr = phash256_arr * 2.0 - 1.0
    phash256_arr = phash256_arr.transpose(2, 0, 1)
    phash256_tensor = torch.tensor(phash256_arr, dtype=torch.float32).unsqueeze(0)
    return phash256_tensor.to(phash256_device)


def _phash256_save_tensor_image(
    phash256_tensor: torch.Tensor,
    phash256_path: str | os.PathLike[str],
) -> None:
    phash256_img = phash256_tensor.detach().cpu().squeeze(0)
    phash256_img = ((phash256_img + 1.0) / 2.0).clamp(0.0, 1.0)
    phash256_arr = (
        phash256_img.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
    )
    Image.fromarray(phash256_arr, mode="RGB").save(phash256_path)


def _phash256_hamming_distance(
    phash256_x: torch.Tensor,
    phash256_y: torch.Tensor,
    phash256_normalize: bool = True,
) -> torch.Tensor:
    phash256_dist = torch.norm(
        phash256_x.float() - phash256_y.float(), p=1, dim=-1
    )
    if phash256_normalize:
        phash256_dist = phash256_dist / phash256_x.shape[-1]
    return phash256_dist


def _phash256_ssim_loss(
    phash256_orig_image: torch.Tensor,
    phash256_optim_image: torch.Tensor,
    phash256_window_size: int = 11,
    phash256_channel: int = 3,
) -> torch.Tensor:
    if torch.all(torch.eq(phash256_orig_image, phash256_optim_image)):
        return torch.tensor(
            0.0,
            dtype=phash256_orig_image.dtype,
            device=phash256_orig_image.device,
        )

    phash256_coords = torch.arange(
        phash256_window_size,
        dtype=phash256_orig_image.dtype,
        device=phash256_orig_image.device,
    )
    phash256_center = phash256_window_size // 2
    phash256_gauss = torch.exp(
        -((phash256_coords - phash256_center) ** 2) / (2.0 * (1.5**2))
    )
    phash256_gauss = phash256_gauss / phash256_gauss.sum()
    phash256_window_1d = phash256_gauss.unsqueeze(1)
    phash256_window_2d = phash256_window_1d @ phash256_window_1d.t()
    phash256_window = (
        phash256_window_2d.float()
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(phash256_channel, 1, phash256_window_size, phash256_window_size)
        .contiguous()
    )

    phash256_mu1 = F.conv2d(
        phash256_orig_image,
        phash256_window,
        padding=phash256_window_size // 2,
        groups=phash256_channel,
    )
    phash256_mu2 = F.conv2d(
        phash256_optim_image,
        phash256_window,
        padding=phash256_window_size // 2,
        groups=phash256_channel,
    )

    phash256_mu1_sq = phash256_mu1.pow(2)
    phash256_mu2_sq = phash256_mu2.pow(2)
    phash256_mu1_mu2 = phash256_mu1 * phash256_mu2

    phash256_sigma1_sq = (
        F.conv2d(
            phash256_orig_image * phash256_orig_image,
            phash256_window,
            padding=phash256_window_size // 2,
            groups=phash256_channel,
        )
        - phash256_mu1_sq
    )
    phash256_sigma2_sq = (
        F.conv2d(
            phash256_optim_image * phash256_optim_image,
            phash256_window,
            padding=phash256_window_size // 2,
            groups=phash256_channel,
        )
        - phash256_mu2_sq
    )
    phash256_sigma12 = (
        F.conv2d(
            phash256_orig_image * phash256_optim_image,
            phash256_window,
            padding=phash256_window_size // 2,
            groups=phash256_channel,
        )
        - phash256_mu1_mu2
    )

    phash256_c1 = 0.01**2
    phash256_c2 = 0.03**2
    phash256_ssim_map = (
        (2 * phash256_mu1_mu2 + phash256_c1)
        * (2 * phash256_sigma12 + phash256_c2)
    ) / (
        (phash256_mu1_sq + phash256_mu2_sq + phash256_c1)
        * (phash256_sigma1_sq + phash256_sigma2_sq + phash256_c2)
    )
    return phash256_ssim_map.mean()


def _phash256_soft_hamming_loss(
    phash256_soft_hash: torch.Tensor,
    phash256_orig_hash_bin: torch.Tensor,
) -> torch.Tensor:
    return -torch.mean(torch.abs(phash256_soft_hash - phash256_orig_hash_bin.float()))


def run_evasion(
    image_path: str,
    threshold: float,
    max_steps: int = 2000,
    device: str = "cpu",
) -> Dict[str, object]:
    phash256_device = torch.device(device)
    phash256_source = _phash256_load_and_preprocess_img(
        image_path, phash256_device, phash256_resize=True
    )
    phash256_orig_image = phash256_source.clone()
    phash256_image_id = Path(image_path).stem

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as phash256_tmp_orig:
        phash256_orig_eval_path = phash256_tmp_orig.name
    _phash256_save_tensor_image(phash256_source, phash256_orig_eval_path)
    try:
        (
            phash256_orig_hash_bin,
            phash256_orig_hash_hex,
            phash256_median_ref,
            _,
        ) = _phash256_hard_details(phash256_orig_eval_path)
    finally:
        if os.path.exists(phash256_orig_eval_path):
            os.remove(phash256_orig_eval_path)

    phash256_orig_hash_bin = phash256_orig_hash_bin.to(phash256_device)

    phash256_source.requires_grad = True
    phash256_optimizer = torch.optim.Adam(params=[phash256_source], lr=1e-3)

    phash256_final_result: Dict[str, object] | None = None
    phash256_attack_start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as phash256_tmp_curr:
        phash256_curr_eval_path = phash256_tmp_curr.name

    try:
        for phash256_step_idx in range(max_steps):
            with torch.no_grad():
                phash256_source.data = torch.clamp(phash256_source.data, min=-1.0, max=1.0)

            phash256_soft_hash = compute_phash256_soft(
                phash256_source,
                median_ref=phash256_median_ref,
                tau=20,
            )
            phash256_target_loss = _phash256_soft_hamming_loss(
                phash256_soft_hash,
                phash256_orig_hash_bin,
            )
            phash256_visual_loss = -_phash256_ssim_loss(
                phash256_orig_image,
                phash256_source,
            )

            phash256_optimizer.zero_grad()
            phash256_total_loss = (
                phash256_target_loss + (0.99**phash256_step_idx) * 5.0 * phash256_visual_loss
            )
            phash256_total_loss.backward()
            phash256_optimizer.step()

            with torch.no_grad():
                _phash256_save_tensor_image(phash256_source, phash256_curr_eval_path)
                (
                    phash256_source_hash_bin,
                    phash256_source_hash_hex,
                ) = compute_phash256_hard(phash256_curr_eval_path)
                phash256_current_img = _phash256_load_and_preprocess_img(
                    phash256_curr_eval_path,
                    phash256_device,
                    phash256_resize=True,
                )

                phash256_dist_raw = _phash256_hamming_distance(
                    phash256_source_hash_bin.unsqueeze(0),
                    phash256_orig_hash_bin.cpu().unsqueeze(0),
                    phash256_normalize=False,
                )
                phash256_dist_norm = _phash256_hamming_distance(
                    phash256_source_hash_bin.unsqueeze(0),
                    phash256_orig_hash_bin.cpu().unsqueeze(0),
                    phash256_normalize=True,
                )
                phash256_l2 = torch.norm(
                    ((phash256_current_img + 1.0) / 2.0)
                    - ((phash256_orig_image + 1.0) / 2.0),
                    p=2,
                )
                phash256_l_inf = torch.norm(
                    ((phash256_current_img + 1.0) / 2.0)
                    - ((phash256_orig_image + 1.0) / 2.0),
                    p=float("inf"),
                )
                phash256_ssim = _phash256_ssim_loss(
                    (phash256_current_img + 1.0) / 2.0,
                    (phash256_orig_image + 1.0) / 2.0,
                )
                phash256_elapsed_ms = (time.perf_counter() - phash256_attack_start) * 1000.0
                phash256_success = int(
                    (phash256_source_hash_hex != phash256_orig_hash_hex)
                    and (float(phash256_dist_norm.item()) >= float(threshold))
                )

                phash256_final_result = {
                    "image_id": phash256_image_id,
                    "threshold": float(threshold),
                    "success": phash256_success,
                    "steps": int(phash256_step_idx + 1),
                    "dist_norm": float(phash256_dist_norm.item()),
                    "dist_raw": int(round(float(phash256_dist_raw.item()))),
                    "l2": float(phash256_l2.item()),
                    "l_inf": float(phash256_l_inf.item()),
                    "ssim": float(phash256_ssim.item()),
                    "time_ms": float(phash256_elapsed_ms),
                    "orig_hash_hex": phash256_orig_hash_hex,
                    "final_hash_hex": phash256_source_hash_hex,
                }

                if phash256_success == 1:
                    break

        if phash256_final_result is None:
            raise RuntimeError("pHash-256 evasion did not produce a final result.")
        return phash256_final_result
    finally:
        if os.path.exists(phash256_curr_eval_path):
            os.remove(phash256_curr_eval_path)


if __name__ == "__main__":
    phash256_repo_root = Path(__file__).resolve().parent.parent
    phash256_input_dir = phash256_repo_root / "inputs" / "inputs_500"
    phash256_first_image = sorted(
        phash256_path
        for phash256_path in phash256_input_dir.iterdir()
        if phash256_path.is_file()
    )[0]
    phash256_result = run_evasion(
        image_path=str(phash256_first_image),
        threshold=0.10,
        max_steps=2000,
        device="cpu",
    )
    print(phash256_result)
