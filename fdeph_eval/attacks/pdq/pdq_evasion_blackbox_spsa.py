"""
Isolated PDQ black-box evasion attack using the exact ``pdqhash`` hard oracle.

This keeps the current repo code untouched while fixing the main SPSA issue:
the perturbed queries are binarized using each image's current median, which
matches PDQ's actual hard-hash decision rule.
"""

import argparse
import concurrent.futures
import os
import threading
import time
from os.path import isfile, join
from random import randint

import numpy as np
import torch
import torchvision.transforms as T
from skimage import feature

from fdeph_eval.utils.structured_logger import StructuredCSVLogger
from losses.quality_losses import ssim_loss
from metrics.hamming_distance import hamming_distance
from utils.image_processing import load_and_preprocess_img, save_images
from utils.pdq_torch import compute_pdq_float, compute_pdq_hard, tensor_to_numpy_rgb

images_lock = threading.Lock()


def compute_pdq_hard_from_numpy(image_np: np.ndarray):
    float_vec, quality = compute_pdq_float(image_np)
    float_vec = np.asarray(float_vec, dtype=np.float32)
    median_val = float(np.median(float_vec))
    hard_hash = torch.tensor((float_vec > median_val).astype(np.float32))
    return hard_hash, median_val, quality


def optimization_thread(url_list, device, step_logger, args):
    thread_id = randint(1, 10000000)
    temp_img = f"curr_pdq_bb_{thread_id}"

    while True:
        with images_lock:
            if not url_list:
                break
            img = url_list.pop(0)

        print("Thread working on " + img)
        source = load_and_preprocess_img(img, device, resize=True)
        input_file_name = os.path.splitext(os.path.basename(img))[0]

        if args.output_folder != "":
            save_images(source, args.output_folder, f"{input_file_name}")

        orig_image = source.clone()
        with torch.no_grad():
            orig_hash_bin, orig_hash_hex, _ = compute_pdq_hard(source)

        edge_mask = None
        if args.edges_only:
            transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])
            image_gray = transform(source.squeeze()).squeeze().cpu().numpy()
            edges = feature.canny(image_gray, sigma=3).astype(int)
            edge_mask = torch.from_numpy(edges).to(device)

        print(f"\nStart optimizing on {img}")
        attack_start = time.perf_counter()
        best_dist_norm = 0.0
        c = args.c_spsa

        for i in range(args.max_steps):
            with torch.no_grad():
                source.data = source.data.clamp(-1, 1)

            spsa_grad_accum = torch.zeros_like(source)
            for _ in range(args.spsa_samples):
                z = torch.sign(torch.randn_like(source))
                if args.edges_only and edge_mask is not None:
                    z = z * edge_mask

                src_plus_np = tensor_to_numpy_rgb((source + c * z).clamp(-1, 1))
                src_minus_np = tensor_to_numpy_rgb((source - c * z).clamp(-1, 1))

                hash_plus_bin, _, _ = compute_pdq_hard_from_numpy(src_plus_np)
                hash_minus_bin, _, _ = compute_pdq_hard_from_numpy(src_minus_np)

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

                scalar = torch.tensor((f_plus - f_minus) / (2 * c), device=source.device)
                spsa_grad_accum += scalar.float() * z

            spsa_grad = spsa_grad_accum / args.spsa_samples

            source_for_ssim = source.detach().requires_grad_(True)
            ssim_val = ssim_loss(orig_image, source_for_ssim)
            if ssim_val.requires_grad:
                ssim_val.backward()
                ssim_grad = source_for_ssim.grad.detach().clone()
            else:
                ssim_grad = torch.zeros_like(source)

            total_grad = -spsa_grad + args.ssim_weight * (0.99 ** i) * ssim_grad
            with torch.no_grad():
                source.data -= args.lr * total_grad

            if i % args.check_interval == 0:
                with torch.no_grad():
                    elapsed_ms = (time.perf_counter() - attack_start) * 1000.0

                    save_images(source, "./temp", temp_img)
                    current_img = load_and_preprocess_img(
                        f"./temp/{temp_img}.png", device, resize=True
                    )
                    source_hash_bin, source_hash_hex, _ = compute_pdq_hard(current_img)

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
                    dist_norm_val = float(dist_norm.item())
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
                            "dist_raw": float(dist_raw.item()),
                            "dist_norm": dist_norm_val,
                            "best_dist_norm": best_dist_norm,
                            "l2": float(l2_distance.item()),
                            "linf": float(linf_distance.item()),
                            "ssim": float(ssim_distance.item()),
                            "success": success,
                            "source_path": img,
                        }
                    )
                    if success == 1:
                        if args.output_folder != "":
                            save_images(source, args.output_folder, f"{input_file_name}_opt_pdq")
                        print(
                            f"Finishing after {i+1} steps - "
                            f"d_raw={float(dist_raw.item()):.1f} - "
                            f"d_norm={dist_norm_val:.4f}"
                        )
                        break

    temp_path = f"./temp/{temp_img}.png"
    if os.path.exists(temp_path):
        os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(description="PDQ black-box SPSA evasion attack.")
    parser.add_argument("--source", type=str, default="./inputs/pdq_subset_10")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--c_spsa", type=float, default=0.02)
    parser.add_argument("--spsa_samples", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--ssim_weight", type=float, default=1.0)
    parser.add_argument("--output_folder", type=str, default="./evasion_attack_outputs_pdq_blackbox")
    parser.add_argument("--edges_only", action="store_true")
    parser.add_argument("--sample_limit", type=int, default=10)
    parser.add_argument("--hamming", type=float, default=0.08)
    parser.add_argument("--threads", dest="num_threads", type=int, default=1)
    parser.add_argument("--check_interval", type=int, default=1)
    parser.add_argument(
        "--step_log_csv",
        type=str,
        default="./logs/attack_steps_pdq_blackbox_subset10.csv",
    )
    parser.add_argument("--hash_method", type=str, default="pdq")
    parser.add_argument("--attack_type", type=str, default="evasion_blackbox_pdq")
    args = parser.parse_args()

    os.makedirs("./temp", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.step_log_csv)), exist_ok=True)
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    step_header = [
        "image_id", "hash_method", "attack_type",
        "step", "elapsed_ms",
        "dist_raw", "dist_norm", "best_dist_norm",
        "l2", "linf", "ssim",
        "success", "source_path",
    ]
    step_logger = StructuredCSVLogger(args.step_log_csv, step_header)
    step_logger.log_row(
        {
            "image_id": "__HYPERPARAMS__",
            "hash_method": args.hash_method,
            "attack_type": args.attack_type,
            "step": 0,
            "source_path": (
                f"source={args.source}; lr={args.lr}; c_spsa={args.c_spsa}; "
                f"spsa_samples={args.spsa_samples}; max_steps={args.max_steps}; "
                f"ssim_w={args.ssim_weight}; hamming_T={args.hamming}; "
                f"threads={args.num_threads}; oracle=current_image_median"
            ),
        }
    )

    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        images = sorted(
            [join(args.source, f) for f in os.listdir(args.source) if isfile(join(args.source, f))]
        )
    else:
        raise RuntimeError(f"{args.source} is neither a file nor a directory.")
    images = images[: args.sample_limit]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        list(executor.map(lambda _: optimization_thread(images, device, step_logger, args), range(args.num_threads)))


if __name__ == "__main__":
    main()
