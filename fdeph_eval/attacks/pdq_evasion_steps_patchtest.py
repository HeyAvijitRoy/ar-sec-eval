"""
Author: Avijit Roy
Project: FDEPH — Security Evaluation of Perceptual Image Hashing

PATCH TEST VERSION
PDQ (Meta/Facebook) evasion attack using SPSA gradient estimation.

Patch purpose:
- Fix oracle mismatch in SPSA:
  use CURRENT median for x+ and x- PDQ binarization,
  matching the real PDQ hard-hash decision rule.
- Add debug logging fields for small pilot runs.

This file is meant for test runs only.
Do not replace the original file yet.
"""
import argparse
import os
from os.path import isfile, join
from random import randint

import numpy as np
import torch
import torchvision.transforms as T
from skimage import feature

from utils.pdq_torch import compute_pdq_hard, compute_pdq_float, tensor_to_numpy_rgb
from losses.quality_losses import ssim_loss
from utils.image_processing import load_and_preprocess_img, save_images
from fdeph_eval.utils.structured_logger import StructuredCSVLogger
from metrics.hamming_distance import hamming_distance
import threading
import concurrent.futures
import time

images_lock = threading.Lock()


def compute_pdq_hard_from_numpy(image_np: np.ndarray):
    """
    Compute hard PDQ hash from uint8 RGB numpy image using the SAME rule
    as compute_pdq_hard(): threshold by the CURRENT image median.
    """
    float_vec, quality = compute_pdq_float(image_np)
    float_vec = np.array(float_vec, dtype=np.float32)
    median_val = float(np.median(float_vec))
    hard_hash = torch.tensor((float_vec > median_val).astype(np.float32))
    return hard_hash, median_val, quality


def optimization_thread(url_list, device, step_logger, args):
    id_ = randint(1, 10000000)
    temp_img = f"curr_image_{id_}"

    while True:
        with images_lock:
            if not url_list:
                break
            img = url_list.pop(0)

        print("Thread working on " + img)

        source = load_and_preprocess_img(img, device, resize=True)
        input_file_name = img.rsplit(sep="/", maxsplit=1)[1].split(".")[0]

        if args.output_folder != "":
            save_images(source, args.output_folder, f"{input_file_name}")

        orig_image = source.clone()

        # ---- Compute original PDQ ONCE -------------------------------------
        with torch.no_grad():
            orig_hash_bin, orig_hash_hex, median_orig = compute_pdq_hard(source)

        # ---- Edge mask (optional) ------------------------------------------
        edge_mask = None
        if args.edges_only:
            transform = T.Compose(
                [T.ToPILImage(), T.Grayscale(), T.ToTensor()]
            )
            image_gray = transform(source.squeeze()).squeeze()
            image_gray = image_gray.cpu().numpy()
            edges = feature.canny(image_gray, sigma=3).astype(int)
            edge_mask = torch.from_numpy(edges).to(device)

        # ---- Optimization cycle (SPSA) -------------------------------------
        print(f"\nStart optimizing on {img}")
        c = args.c_spsa
        attack_start = time.perf_counter()

        best_dist_norm = 0.0

        for i in range(args.max_steps):
            # Clamp to valid range
            with torch.no_grad():
                source.data = source.data.clamp(-1, 1)

            # ---- SPSA gradient estimate ------------------------------------
            spsa_grad_accum = torch.zeros_like(source)

            for _ in range(args.spsa_samples):
                z = torch.sign(torch.randn_like(source))  # Rademacher sample

                if args.edges_only and edge_mask is not None:
                    z = z * edge_mask

                src_plus = (source + c * z).clamp(-1, 1)
                src_minus = (source - c * z).clamp(-1, 1)

                src_plus_np = tensor_to_numpy_rgb(src_plus)
                src_minus_np = tensor_to_numpy_rgb(src_minus)

                # PATCH: use CURRENT median for each perturbed image
                hash_plus_bin, median_plus, _ = compute_pdq_hard_from_numpy(src_plus_np)
                hash_minus_bin, median_minus, _ = compute_pdq_hard_from_numpy(src_minus_np)

                f_plus = float(
                    hamming_distance(
                        hash_plus_bin.unsqueeze(0),
                        orig_hash_bin.unsqueeze(0),
                        normalize=True,
                    )
                )
                f_minus = float(
                    hamming_distance(
                        hash_minus_bin.unsqueeze(0),
                        orig_hash_bin.unsqueeze(0),
                        normalize=True,
                    )
                )

                scalar = torch.tensor((f_plus - f_minus) / (2 * c), device=source.device).float()
                spsa_grad_accum += scalar * z

            spsa_grad = spsa_grad_accum / args.spsa_samples

            # ---- SSIM gradient ---------------------------------------------
            source_for_ssim = source.detach().requires_grad_(True)
            ssim_val = ssim_loss(orig_image, source_for_ssim)
            if ssim_val.requires_grad:
                ssim_val.backward()
                ssim_grad = source_for_ssim.grad.detach().clone()
                source_for_ssim.grad = None
            else:
                ssim_grad = torch.zeros_like(source)

            # ---- Combined update -------------------------------------------
            # Maximize hash distance, minimize SSIM loss
            total_grad = (
                -spsa_grad
                + args.ssim_weight * (0.99 ** i) * ssim_grad
            )

            with torch.no_grad():
                source.data -= args.lr * total_grad

            # ---- Check & log -----------------------------------------------
            if i % args.check_interval == 0:
                with torch.no_grad():
                    elapsed_ms = (time.perf_counter() - attack_start) * 1000.0

                    # Keep round-trip behavior identical to your baseline script
                    save_images(source, "./temp", temp_img)
                    current_img = load_and_preprocess_img(
                        f"./temp/{temp_img}.png", device, resize=True
                    )

                    # Real PDQ hard hash on current reloaded image
                    source_hash_bin, source_hash_hex, current_median = compute_pdq_hard(current_img)

                    dist_raw = hamming_distance(
                        source_hash_bin.unsqueeze(0),
                        orig_hash_bin.unsqueeze(0),
                        normalize=False,
                    )
                    dist_norm = hamming_distance(
                        source_hash_bin.unsqueeze(0),
                        orig_hash_bin.unsqueeze(0),
                        normalize=True,
                    )

                    dist_raw_val = float(dist_raw.item())
                    dist_norm_val = float(dist_norm.item())
                    flipped_bits = int(round(dist_norm_val * 256))

                    improved = 1 if dist_norm_val > best_dist_norm else 0
                    if dist_norm_val > best_dist_norm:
                        best_dist_norm = dist_norm_val

                    l2_distance = torch.norm(
                        ((current_img + 1) / 2) - ((orig_image + 1) / 2), p=2
                    )
                    linf_distance = torch.norm(
                        ((current_img + 1) / 2) - ((orig_image + 1) / 2),
                        p=float("inf"),
                    )
                    ssim_distance = ssim_loss(
                        (current_img + 1) / 2, (orig_image + 1) / 2
                    )

                    grad_abs_mean = float(spsa_grad.abs().mean().item())

                    success = int(
                        (source_hash_hex != orig_hash_hex)
                        and (dist_norm_val >= args.hamming)
                    )

                    step_logger.log_row(
                        {
                            "image_id": input_file_name,
                            "hash_method": args.hash_method,
                            "attack_type": args.attack_type,
                            "step": i + 1,
                            "elapsed_ms": round(elapsed_ms, 3),
                            "dist_raw": dist_raw_val,
                            "dist_norm": dist_norm_val,
                            "best_dist_norm": float(best_dist_norm),
                            "flipped_bits": flipped_bits,
                            "improved": improved,
                            "grad_abs_mean": grad_abs_mean,
                            "l2": float(l2_distance.item()),
                            "linf": float(linf_distance.item()),
                            "ssim": float(ssim_distance.item()),
                            "success": success,
                            "source_path": img,
                        }
                    )

                    if success == 1:
                        if args.output_folder != "":
                            save_images(
                                source,
                                args.output_folder,
                                f"{input_file_name}_opt",
                            )
                        print(
                            f"Finishing after {i+1} steps - "
                            f"L2: {float(l2_distance.item()):.4f} - "
                            f"LInf: {float(linf_distance.item()):.4f} - "
                            f"SSIM: {float(ssim_distance.item()):.4f} - "
                            f"d_raw: {dist_raw_val:.4f} - "
                            f"d_norm: {dist_norm_val:.4f} - "
                            f"best: {best_dist_norm:.4f} - "
                            f"flipped_bits: {flipped_bits}"
                        )
                        break

    temp_path = f"./temp/{temp_img}.png"
    if os.path.exists(temp_path):
        os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Perform PDQ evasion attack via SPSA (patch test version)."
    )
    parser.add_argument(
        "--source", dest="source", type=str,
        default="inputs/source.png",
        help="Image or directory of images to manipulate",
    )
    parser.add_argument(
        "--lr", dest="lr", default=0.02,
        type=float, help="SPSA step size",
    )
    parser.add_argument(
        "--learning_rate", dest="lr", default=argparse.SUPPRESS,
        type=float, help="Alias for --lr (backward compatibility)",
    )
    parser.add_argument(
        "--c_spsa", dest="c_spsa", default=0.02,
        type=float, help="SPSA perturbation magnitude",
    )
    parser.add_argument(
        "--spsa_samples", dest="spsa_samples", default=4,
        type=int, help="Number of SPSA estimates to average per step",
    )
    parser.add_argument(
        "--max_steps", dest="max_steps", default=1000,
        type=int, help="Maximum number of SPSA optimisation steps",
    )
    parser.add_argument(
        "--optimizer", dest="optimizer", default="SPSA",
        type=str, help="Optimizer label (ignored; SPSA is always used)",
    )
    parser.add_argument(
        "--ssim_weight", dest="ssim_weight", default=1.0,
        type=float, help="Weight of SSIM visual-quality regulariser",
    )
    parser.add_argument(
        "--experiment_name", dest="experiment_name",
        default="pdq_evasion_patchtest", type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--output_folder", dest="output_folder",
        default="evasion_attack_outputs_pdq_patchtest", type=str,
        help="Folder to save optimised images",
    )
    parser.add_argument(
        "--edges_only", dest="edges_only",
        action="store_true",
        help="Restrict perturbation to edge pixels only",
    )
    parser.add_argument(
        "--sample_limit", dest="sample_limit",
        default=10, type=int,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--hamming", dest="hamming",
        default=0.10, type=float,
        help="Minimum normalised Hamming distance threshold to declare success",
    )
    parser.add_argument(
        "--threads", dest="num_threads",
        default=1, type=int,
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--check_interval", dest="check_interval",
        default=1, type=int,
        help="Number of optimisation steps between hash-change checks",
    )
    parser.add_argument(
        "--step_log_csv", dest="step_log_csv",
        default="./logs/attack_steps_pdq_evasion_patchtest.csv", type=str,
        help="CSV path for step-by-step logging",
    )
    parser.add_argument(
        "--hash_method", dest="hash_method",
        default="pdq", type=str,
        help="Hash method label for logs",
    )
    parser.add_argument(
        "--attack_type", dest="attack_type",
        default="evasion", type=str,
        help="Attack type label for logs",
    )
    args = parser.parse_args()

    os.makedirs("./temp", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.output_folder != "":
        os.makedirs(args.output_folder, exist_ok=True)

    step_header = [
        "image_id", "hash_method", "attack_type",
        "step", "elapsed_ms",
        "dist_raw", "dist_norm", "best_dist_norm",
        "flipped_bits", "improved", "grad_abs_mean",
        "l2", "linf", "ssim",
        "success",
        "source_path",
    ]
    step_logger = StructuredCSVLogger(args.step_log_csv, step_header)

    step_logger.log_row(
        {
            "image_id": "__HYPERPARAMS__",
            "hash_method": args.hash_method,
            "attack_type": args.attack_type,
            "step": 0,
            "elapsed_ms": 0,
            "dist_raw": "",
            "dist_norm": "",
            "best_dist_norm": "",
            "flipped_bits": "",
            "improved": "",
            "grad_abs_mean": "",
            "l2": "",
            "linf": "",
            "ssim": "",
            "success": "",
            "source_path": (
                f"source={args.source}; lr={args.lr}; c_spsa={args.c_spsa}; "
                f"spsa_samples={args.spsa_samples}; max_steps={args.max_steps}; "
                f"ssim_w={args.ssim_weight}; edges_only={args.edges_only}; "
                f"hamming_T={args.hamming}; check_interval={args.check_interval}; "
                f"threads={args.num_threads}; PATCH=current_median_spsa"
            ),
        }
    )

    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        images = sorted(
            [
                join(args.source, f)
                for f in os.listdir(args.source)
                if isfile(join(args.source, f))
            ]
        )
    else:
        raise RuntimeError(f"{args.source} is neither a file nor a directory.")

    images = images[: args.sample_limit]

    def thread_function(_):
        return optimization_thread(images, device, step_logger, args)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_threads
    ) as executor:
        list(executor.map(thread_function, range(args.num_threads)))

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()