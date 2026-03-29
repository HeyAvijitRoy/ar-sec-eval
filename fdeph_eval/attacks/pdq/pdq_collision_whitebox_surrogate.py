"""
Isolated PDQ white-box targeted collision attack driven by the PDQ surrogate.

This keeps the current repo structure intact by living entirely under
``fdeph_eval/attacks/pdq/`` and reusing the exact PDQ hard hash only for
target selection and success checks.
"""

import argparse
import concurrent.futures
import csv
import os
import pathlib
import sys
import threading
import time
from os.path import isfile, join
from random import randint

import torch
import torch.nn.functional as F

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fdeph_eval.attacks.pdq.pdq_surrogate import compute_pdq_soft_surrogate
from fdeph_eval.utils.structured_logger import StructuredCSVLogger
from losses.quality_losses import ssim_loss
from metrics.hamming_distance import hamming_distance
from utils.image_processing import load_and_preprocess_img, save_images
from utils.pdq_torch import compute_pdq_hard

images_lock = threading.Lock()


def collision_target_loss(soft_hash: torch.Tensor, target_hash: torch.Tensor) -> torch.Tensor:
    soft_hash = soft_hash.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(soft_hash, target_hash.float())


def load_target_hashset(csv_path: str, device: str):
    target_entries = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_path = row["image"]
            hash_bin_str = row["hash_bin"]
            hash_hex_str = row["hash_hex"]
            target_entries.append(
                {
                    "image_path": image_path,
                    "hash_bin": torch.tensor(
                        [int(b) for b in hash_bin_str], dtype=torch.float32, device=device
                    ),
                    "hash_bin_str": hash_bin_str,
                    "hash_hex": hash_hex_str,
                }
            )
    if not target_entries:
        raise RuntimeError(f"No targets found in {csv_path}")
    return target_entries


def choose_target(source_path: str, source_hash_bin: torch.Tensor, source_hash_hex: str, target_entries):
    source_abs = os.path.abspath(source_path)
    best = None
    best_dist = None

    for entry in target_entries:
        entry_abs = os.path.abspath(entry["image_path"])
        if entry_abs == source_abs:
            continue
        if entry["hash_hex"] == source_hash_hex:
            continue

        dist = float(
            hamming_distance(
                source_hash_bin.unsqueeze(0),
                entry["hash_bin"].unsqueeze(0),
                normalize=True,
            ).item()
        )
        if best is None or dist < best_dist:
            best = entry
            best_dist = dist

    if best is None:
        raise RuntimeError(
            "Could not find a distinct collision target. "
            "Use a target hashset with different images and hashes."
        )
    return best, best_dist


def optimization_thread(url_list, device, step_logger, args, target_entries):
    thread_id = randint(1, 10000000)
    temp_img = f"curr_pdq_collision_{thread_id}"

    while True:
        with images_lock:
            if not url_list:
                break
            img = url_list.pop(0)

        print("Thread working on " + img)
        source = load_and_preprocess_img(img, device, resize=True)
        input_file_name = os.path.splitext(os.path.basename(img))[0]

        if args.output_folder:
            save_images(source, args.output_folder, f"{input_file_name}")

        orig_image = source.clone()
        with torch.no_grad():
            official_source_hash, source_hash_hex, _ = compute_pdq_hard(source)
            _, _, surrogate_source_median = compute_pdq_soft_surrogate(source)
            target_entry, initial_target_dist = choose_target(
                img, official_source_hash, source_hash_hex, target_entries
            )
            target_hash = target_entry["hash_bin"]
            target_image_path = target_entry["image_path"]
            target_hash_hex = target_entry["hash_hex"]
            target_image = load_and_preprocess_img(target_image_path, device, resize=True)

        source.requires_grad = True
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(params=[source], lr=args.learning_rate)
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(params=[source], lr=args.learning_rate)
        else:
            raise RuntimeError(f"Unsupported optimizer: {args.optimizer}")

        print(f"\nStart PDQ collision optimization on {img}")
        print(f"  Target hash : {target_hash_hex}")
        print(f"  Target image: {target_image_path}")
        attack_start = time.perf_counter()
        best_target_dist_norm = initial_target_dist

        for i in range(args.max_steps):
            with torch.no_grad():
                source.data = source.data.clamp(-1, 1)

            soft_hash, _, _ = compute_pdq_soft_surrogate(
                source,
                median_ref=surrogate_source_median,
                temperature=args.temperature,
            )
            target_loss = collision_target_loss(soft_hash, target_hash)
            visual_loss = -ssim_loss(orig_image, source)
            total_loss = target_loss + (0.99 ** i) * args.ssim_weight * visual_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % args.check_interval == 0:
                with torch.no_grad():
                    elapsed_ms = (time.perf_counter() - attack_start) * 1000.0
                    save_images(source, "./temp", temp_img)
                    current_img = load_and_preprocess_img(
                        f"./temp/{temp_img}.png", device, resize=True
                    )
                    current_hash_bin, current_hash_hex, _ = compute_pdq_hard(current_img)

                    target_dist_raw = hamming_distance(
                        current_hash_bin.unsqueeze(0),
                        target_hash.unsqueeze(0),
                        normalize=False,
                    )
                    target_dist_norm = hamming_distance(
                        current_hash_bin.unsqueeze(0),
                        target_hash.unsqueeze(0),
                        normalize=True,
                    )
                    target_dist_norm_val = float(target_dist_norm.item())
                    if target_dist_norm_val < best_target_dist_norm:
                        best_target_dist_norm = target_dist_norm_val

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

                    success = int(target_dist_norm_val <= args.target_hamming)
                    step_logger.log_row(
                        {
                            "image_id": input_file_name,
                            "hash_method": args.hash_method,
                            "attack_type": args.attack_type,
                            "target_image_id": os.path.splitext(os.path.basename(target_image_path))[0],
                            "target_hash_hex": target_hash_hex,
                            "step": i + 1,
                            "elapsed_ms": round(elapsed_ms, 3),
                            "target_dist_raw": float(target_dist_raw.item()),
                            "target_dist_norm": target_dist_norm_val,
                            "best_target_dist_norm": best_target_dist_norm,
                            "l2": float(l2_distance.item()),
                            "linf": float(linf_distance.item()),
                            "ssim": float(ssim_distance.item()),
                            "success": success,
                            "source_path": img,
                            "target_path": target_image_path,
                        }
                    )
                    if success == 1:
                        if args.output_folder:
                            save_images(source, args.output_folder, f"{input_file_name}_opt_pdq_collision")
                            save_images(target_image, args.output_folder, f"{input_file_name}_target_pdq_collision")
                        print(
                            f"Finishing after {i+1} steps - "
                            f"target_d_raw={float(target_dist_raw.item()):.1f} - "
                            f"target_d_norm={target_dist_norm_val:.4f}"
                        )
                        break

    temp_path = f"./temp/{temp_img}.png"
    if os.path.exists(temp_path):
        os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(description="PDQ white-box surrogate collision attack.")
    parser.add_argument("--source", type=str, default="./inputs/pdq_subset_10")
    parser.add_argument("--target_hashset", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--ssim_weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--output_folder", type=str, default="./collision_attack_outputs_pdq_whitebox")
    parser.add_argument("--sample_limit", type=int, default=10)
    parser.add_argument(
        "--target_hamming",
        type=float,
        default=0.0,
        help="Success threshold on normalized Hamming distance to the target hash. 0.0 means exact collision.",
    )
    parser.add_argument("--threads", dest="num_threads", type=int, default=2)
    parser.add_argument("--check_interval", type=int, default=1)
    parser.add_argument(
        "--step_log_csv",
        type=str,
        default="./logs/attack_steps_pdq_collision_whitebox_subset10.csv",
    )
    parser.add_argument("--hash_method", type=str, default="pdq")
    parser.add_argument("--attack_type", type=str, default="collision_whitebox_pdq")
    args = parser.parse_args()

    os.makedirs("./temp", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.step_log_csv)), exist_ok=True)
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target_entries = load_target_hashset(args.target_hashset, device)

    step_header = [
        "image_id", "hash_method", "attack_type",
        "target_image_id", "target_hash_hex",
        "step", "elapsed_ms",
        "target_dist_raw", "target_dist_norm", "best_target_dist_norm",
        "l2", "linf", "ssim",
        "success", "source_path", "target_path",
    ]
    step_logger = StructuredCSVLogger(args.step_log_csv, step_header)
    step_logger.log_row(
        {
            "image_id": "__HYPERPARAMS__",
            "hash_method": args.hash_method,
            "attack_type": args.attack_type,
            "step": 0,
            "source_path": (
                f"source={args.source}; target_hashset={args.target_hashset}; "
                f"lr={args.learning_rate}; opt={args.optimizer}; "
                f"ssim_w={args.ssim_weight}; temperature={args.temperature}; "
                f"max_steps={args.max_steps}; target_hamming_T={args.target_hamming}; "
                f"threads={args.num_threads}; target_selection=nearest_distinct_hash; "
                f"surrogate_median=source"
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
        list(
            executor.map(
                lambda _: optimization_thread(images, device, step_logger, args, target_entries),
                range(args.num_threads),
            )
        )


if __name__ == "__main__":
    main()
