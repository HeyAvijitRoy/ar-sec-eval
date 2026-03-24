"""
Author: Avijit Roy
Project: FDEPH — Security Evaluation of Perceptual Image Hashing

PDQ (Meta/Facebook) evasion attack using SPSA gradient estimation.

Derived from:
  fdeph_eval/attacks/phash_evasion_steps.py

Key difference from pHash attack:
  pHash uses a white-box differentiable surrogate (sigmoid over DCT).
  PDQ uses SPSA (zeroth-order) because the Jarosz filter is not
  differentiable through our PyTorch pipeline. This matches the
  black-box threat model evaluated in Madden et al. (2024).

SPSA update rule (per step i):
  z ~ Uniform({-1,+1})^(H*W*3)          random perturbation direction
  x+ = clamp(x + c*z, -1, 1)
  x- = clamp(x - c*z, -1, 1)
  f+ = hamming_distance(PDQ(x+), orig_hash)
  f- = hamming_distance(PDQ(x-), orig_hash)
  g  = (f+ - f-) / (2*c) * z            gradient estimate
  visual_grad = -nabla_ssim(x, orig)     SSIM regularizer (autodiff OK)
  x  = x - lr * (g + ssim_weight * visual_grad)

Two pdqhash.compute_float() calls per step. Cost ≈ 2.6ms/step.
Logging schema identical to nhash and phash attack scripts.
"""
import argparse
import os
from os.path import isfile, join
from random import randint

import numpy as np
import torch
import torchvision.transforms as T
from skimage import feature
from skimage.color import rgb2gray

from utils.pdq_torch import compute_pdq_hard, compute_pdq_float, tensor_to_numpy_rgb
from losses.quality_losses import ssim_loss
from utils.image_processing import load_and_preprocess_img, save_images
from fdeph_eval.utils.structured_logger import StructuredCSVLogger
from metrics.hamming_distance import hamming_distance
import threading
import concurrent.futures
import time

images_lock = threading.Lock()


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

        # ---- Compute original PDQ ONCE (frozen median_orig) ----------------
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
        attack_start = time.perf_counter()  # BEFORE loop

        for i in range(args.max_steps):
            # Clamp to valid range
            with torch.no_grad():
                source.data = source.data.clamp(-1, 1)

            # ---- SPSA gradient estimate (averaged over spsa_samples) -------
            spsa_grad_accum = torch.zeros_like(source)
            for _ in range(args.spsa_samples):
                z = torch.sign(torch.randn_like(source))  # Rademacher sample

                if args.edges_only and edge_mask is not None:
                    z = z * edge_mask

                src_plus_np  = tensor_to_numpy_rgb((source + c * z).clamp(-1, 1))
                src_minus_np = tensor_to_numpy_rgb((source - c * z).clamp(-1, 1))

                float_plus,  _ = compute_pdq_float(src_plus_np)
                float_minus, _ = compute_pdq_float(src_minus_np)

                float_plus  = np.array(float_plus,  dtype=np.float32)
                float_minus = np.array(float_minus, dtype=np.float32)

                hash_plus_bin  = torch.tensor(
                    (float_plus  > median_orig).astype(np.float32)
                )
                hash_minus_bin = torch.tensor(
                    (float_minus > median_orig).astype(np.float32)
                )

                f_plus  = float(hamming_distance(
                    hash_plus_bin.unsqueeze(0),
                    orig_hash_bin.unsqueeze(0),
                    normalize=True,
                ))
                f_minus = float(hamming_distance(
                    hash_minus_bin.unsqueeze(0),
                    orig_hash_bin.unsqueeze(0),
                    normalize=True,
                ))

                spsa_grad_accum += (
                    torch.tensor((f_plus - f_minus) / (2 * c)).float() * z
                )

            spsa_grad = spsa_grad_accum / args.spsa_samples

            # ---- SSIM gradient (differentiable, use autograd) --------------
            source_for_ssim = source.detach().requires_grad_(True)
            ssim_val = ssim_loss(orig_image, source_for_ssim)
            ssim_val.backward()
            ssim_grad = source_for_ssim.grad.detach()

            # ---- Combined update -------------------------------------------
            # Gradient ASCENT on hash distance, descent on SSIM loss
            # We want to INCREASE hash distance, so:
            total_grad = (
                -spsa_grad
                + args.ssim_weight * (0.99 ** i) * ssim_grad
            )

            with torch.no_grad():
                source.data -= args.lr * total_grad

            # ---- Check & log at interval -----------------------------------
            if i % args.check_interval == 0:
                with torch.no_grad():
                    elapsed_ms = (time.perf_counter() - attack_start) * 1000.0

                    # Round-trip through disk (matches nhash/phash behaviour)
                    save_images(source, "./temp", temp_img)
                    current_img = load_and_preprocess_img(
                        f"./temp/{temp_img}.png", device, resize=True
                    )

                    # Compute PDQ on reloaded image
                    source_hash_bin, source_hash_hex, _ = compute_pdq_hard(
                        current_img
                    )

                    # Distances (raw + normalised)
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

                    # Distortion metrics in [0, 1] space
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

                    # Success: hash changed AND normalised Hamming >= threshold
                    success = int(
                        (source_hash_hex != orig_hash_hex)
                        and (dist_norm >= args.hamming)
                    )

                    step_logger.log_row(
                        {
                            "image_id": input_file_name,
                            "hash_method": args.hash_method,
                            "attack_type": args.attack_type,
                            "step": i + 1,
                            "elapsed_ms": round(elapsed_ms, 3),
                            "dist_raw": float(dist_raw.item()),
                            "dist_norm": float(dist_norm.item()),
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
                            f"d_raw: {float(dist_raw.item()):.4f} - "
                            f"d_norm: {float(dist_norm.item()):.4f}"
                        )
                        break

    temp_path = f"./temp/{temp_img}.png"
    if os.path.exists(temp_path):
        os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Perform PDQ evasion attack via SPSA (FDEPH comparative study)."
    )
    parser.add_argument(
        "--source", dest="source", type=str,
        default="inputs/source.png",
        help="Image or directory of images to manipulate",
    )
    # SPSA step size — primary arg; --learning_rate kept for CLI compatibility
    parser.add_argument(
        "--lr", dest="lr", default=0.005,
        type=float, help="SPSA step size",
    )
    parser.add_argument(
        "--learning_rate", dest="lr", default=argparse.SUPPRESS,
        type=float, help="Alias for --lr (backward compatibility)",
    )
    parser.add_argument(
        "--c_spsa", dest="c_spsa", default=0.01,
        type=float, help="SPSA perturbation magnitude",
    )
    parser.add_argument(
        "--spsa_samples", dest="spsa_samples", default=1,
        type=int, help="Number of SPSA gradient estimates to average per step",
    )
    parser.add_argument(
        "--max_steps", dest="max_steps", default=2000,
        type=int, help="Maximum number of SPSA optimisation steps",
    )
    # Kept for CLI compatibility (ignored by SPSA)
    parser.add_argument(
        "--optimizer", dest="optimizer", default="SPSA",
        type=str, help="Optimizer label (ignored; SPSA is always used)",
    )
    parser.add_argument(
        "--ssim_weight", dest="ssim_weight", default=5,
        type=float, help="Weight of the SSIM visual-quality regulariser",
    )
    parser.add_argument(
        "--experiment_name", dest="experiment_name",
        default="pdq_evasion_attack", type=str,
        help="Experiment name (used in output folder naming)",
    )
    parser.add_argument(
        "--output_folder", dest="output_folder",
        default="evasion_attack_outputs_pdq", type=str,
        help="Folder to save optimised images",
    )
    parser.add_argument(
        "--edges_only", dest="edges_only",
        action="store_true",
        help="Restrict perturbation to edge pixels only (applied to z)",
    )
    parser.add_argument(
        "--sample_limit", dest="sample_limit",
        default=10000000, type=int,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--hamming", dest="hamming",
        default=0.10, type=float,
        help="Minimum normalised Hamming distance threshold to declare success",
    )
    parser.add_argument(
        "--threads", dest="num_threads",
        default=4, type=int,
        help="Number of parallel worker threads (SPSA is CPU-bound)",
    )
    parser.add_argument(
        "--check_interval", dest="check_interval",
        default=1, type=int,
        help="Number of optimisation steps between hash-change checks",
    )
    # ---- FDEPH Eval Logging ----
    parser.add_argument(
        "--step_log_csv", dest="step_log_csv",
        default="./logs/attack_steps_pdq_evasion.csv", type=str,
        help="CSV path for step-by-step (long format) logging",
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

    # Create temp folder
    os.makedirs("./temp", exist_ok=True)

    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Prepare output folder
    if args.output_folder != "":
        try:
            os.mkdir(args.output_folder)
        except FileExistsError:
            if not os.listdir(args.output_folder):
                print(f"Folder {args.output_folder} already exists and is empty.")
            else:
                print(f"Folder {args.output_folder} already exists and is not empty.")

    # ---- Step-by-step CSV logger (long format) — identical schema to nhash/phash ----
    step_header = [
        "image_id", "hash_method", "attack_type",
        "step", "elapsed_ms",
        "dist_raw", "dist_norm",
        "l2", "linf", "ssim",
        "success",
        "source_path",
    ]
    step_logger = StructuredCSVLogger(args.step_log_csv, step_header)

    # Hyperparams row (step=0, for reproducibility)
    step_logger.log_row(
        {
            "image_id": "__HYPERPARAMS__",
            "hash_method": args.hash_method,
            "attack_type": args.attack_type,
            "step": 0,
            "elapsed_ms": 0,
            "dist_raw": "",
            "dist_norm": "",
            "l2": "",
            "linf": "",
            "ssim": "",
            "success": "",
            "source_path": (
                f"source={args.source}; lr={args.lr}; c_spsa={args.c_spsa}; "
                f"spsa_samples={args.spsa_samples}; max_steps={args.max_steps}; "
                f"ssim_w={args.ssim_weight}; edges_only={args.edges_only}; "
                f"hamming_T={args.hamming}; check_interval={args.check_interval}; "
                f"threads={args.num_threads}"
            ),
        }
    )

    # Load images
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

    # Launch worker threads (each pops from shared images list)
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
