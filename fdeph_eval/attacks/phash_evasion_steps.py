"""
# Author: Avijit Roy
# Project: FDEPH - Security Evaluation of Perceptual Image Hashing

# pHash parallel attack for FDEPH comparative study.

# Derived from:
#   fdeph_eval/attacks/nhash_evasion_steps.py  (NeuralHash evasion, FAccT 2022 lineage)

# Modifications relative to nhash_evasion_steps.py:
#   - Removed NeuralHash model loading (no model.pth, no NeuralHash class)
#   - Removed load_hash_matrix / seed (not needed for pHash)
#   - Uses compute_phash_soft / compute_phash_hard from utils.phash_torch
#   - Soft Hamming loss: -mean(|soft_hash - orig_hash_bin|) to maximise bit flip
#   - All logging fields, CSV schema, and thread-safety pattern are identical
#     to nhash_evasion_steps.py so results are directly comparable.
"""
import argparse
import os
from os.path import isfile, join
from random import randint

import torch
import torchvision.transforms as T
from skimage import feature
from skimage.color import rgb2gray

from utils.phash_torch import compute_phash_soft, compute_phash_hard
from losses.quality_losses import ssim_loss
from utils.image_processing import load_and_preprocess_img, save_images
from fdeph_eval.utils.structured_logger import StructuredCSVLogger
from metrics.hamming_distance import hamming_distance
import threading
import concurrent.futures
import time

images_lock = threading.Lock()


def soft_hamming_loss(
    soft_hash: torch.Tensor,
    orig_hash_bin: torch.Tensor,
) -> torch.Tensor:
    """
    Surrogate loss that drives bits away from their original values.

    Returns -mean(|soft_hash - orig_hash_bin|).
    Minimising this loss maximises the expected number of flipped bits.
    """
    return -torch.mean(torch.abs(soft_hash - orig_hash_bin.float()))


def optimization_thread(url_list, device, step_logger, args):
    id_ = randint(1, 10000000)
    temp_img = f"curr_image_{id_}"

    while True:
        with images_lock:
            if not url_list:
                break
            img = url_list.pop(0)

        print("Thread working on " + img)

        if args.optimize_original:
            resize = T.Resize((360, 360))
            source = load_and_preprocess_img(img, device, resize=False)
        else:
            source = load_and_preprocess_img(img, device, resize=True)

        input_file_name = img.rsplit(sep="/", maxsplit=1)[1].split(".")[0]

        if args.output_folder != "":
            save_images(source, args.output_folder, f"{input_file_name}")

        orig_image = source.clone()

        # ---- Compute original pHash ONCE (frozen median_ref) ---------------
        with torch.no_grad():
            if args.optimize_original:
                unmodified_hash_bin, unmodified_hash_hex, median_ref = compute_phash_hard(
                    resize(source)
                )
            else:
                unmodified_hash_bin, unmodified_hash_hex, median_ref = compute_phash_hard(
                    source
                )

        # ---- Edge mask (optional) ------------------------------------------
        if args.edges_only:
            transform = T.Compose(
                [T.ToPILImage(), T.Grayscale(), T.ToTensor()]
            )
            image_gray = transform(source.squeeze()).squeeze()
            image_gray = image_gray.cpu().numpy()
            edges = feature.canny(image_gray, sigma=3).astype(int)
            edge_mask = torch.from_numpy(edges).to(device)

        # ---- Optimizer setup -----------------------------------------------
        source.requires_grad = True
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(params=[source], lr=args.learning_rate)
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(params=[source], lr=args.learning_rate)
        else:
            raise RuntimeError(
                f"{args.optimizer} is not a valid optimizer. "
                "Choose from [Adam, SGD]."
            )

        # ---- Optimization cycle --------------------------------------------
        print(f"\nStart optimizing on {img}")
        attack_start = time.perf_counter()   # AR: start timer before loop
        for i in range(10000):
            with torch.no_grad():
                source.data = torch.clamp(source, min=-1, max=1)

            if args.optimize_original:
                soft_hash, _, _ = compute_phash_soft(
                    resize(source),
                    median_ref=median_ref,
                    temperature=args.temperature,
                )
            else:
                soft_hash, _, _ = compute_phash_soft(
                    source,
                    median_ref=median_ref,
                    temperature=args.temperature,
                )

            target_loss = soft_hamming_loss(soft_hash, unmodified_hash_bin)
            visual_loss = -ssim_loss(orig_image, source)

            optimizer.zero_grad()
            total_loss = target_loss + 0.99 ** i * args.ssim_weight * visual_loss
            total_loss.backward()

            if args.edges_only:
                optimizer.param_groups[0]["params"][0].grad *= edge_mask

            optimizer.step()

            # ---- Check & log at interval -----------------------------------
            if i % args.check_interval == 0:
                with torch.no_grad():
                    # # Timing
                    # if i == 0:
                    #     attack_start = time.perf_counter()
                    elapsed_ms = (time.perf_counter() - attack_start) * 1000.0


                    # Round-trip through disk (matches nhash behaviour)
                    save_images(source, "./temp", temp_img)
                    current_img = load_and_preprocess_img(
                        f"./temp/{temp_img}.png", device, resize=True
                    )

                    # Compute pHash on reloaded image
                    source_hash_bin, source_hash_hex, _ = compute_phash_hard(current_img)

                    # Distances (raw + normalised)
                    dist_raw = hamming_distance(
                        source_hash_bin.unsqueeze(0),
                        unmodified_hash_bin.unsqueeze(0),
                        normalize=False,
                    )
                    dist_norm = hamming_distance(
                        source_hash_bin.unsqueeze(0),
                        unmodified_hash_bin.unsqueeze(0),
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
                        (source_hash_hex != unmodified_hash_hex)
                        and (dist_norm >= args.hamming)
                    )

                    # Write one step row (long format, identical schema to nhash)
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
        description="Perform pHash evasion attack (FDEPH comparative study)."
    )
    parser.add_argument(
        "--source", dest="source", type=str,
        default="inputs/source.png",
        help="Image or directory of images to manipulate",
    )
    parser.add_argument(
        "--learning_rate", dest="learning_rate", default=1e-3,
        type=float, help="Step size for the PGD optimisation",
    )
    parser.add_argument(
        "--optimizer", dest="optimizer", default="Adam",
        type=str, help="Optimizer class [Adam | SGD]",
    )
    parser.add_argument(
        "--ssim_weight", dest="ssim_weight", default=5,
        type=float, help="Weight of the SSIM visual-quality loss",
    )
    parser.add_argument(
        "--experiment_name", dest="experiment_name",
        default="phash_evasion_attack", type=str,
        help="Experiment name (used in output folder naming)",
    )
    parser.add_argument(
        "--output_folder", dest="output_folder",
        default="evasion_attack_outputs_phash", type=str,
        help="Folder to save optimised images",
    )
    parser.add_argument(
        "--edges_only", dest="edges_only",
        action="store_true",
        help="Restrict perturbation to edge pixels only",
    )
    parser.add_argument(
        "--optimize_original", dest="optimize_original",
        action="store_true",
        help="Optimise on resized image (pass resized copy to hash)",
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
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--check_interval", dest="check_interval",
        default=1, type=int,
        help="Number of optimisation steps between hash-change checks",
    )
    parser.add_argument(
        "--temperature", dest="temperature",
        # "Temperature T=20 was selected to provide sufficient gradient signal while approximating the hard binarization step. 
        # Empirical results confirm convergence within a median of 13 steps across 500 images, validating this choice."
        default=20.0, type=float,
        help="Sigmoid temperature for the soft pHash surrogate",
    )
    # ---- FDEPH Eval Logging ----
    parser.add_argument(
        "--step_log_csv", dest="step_log_csv",
        default="./logs/attack_steps_phash_evasion.csv", type=str,
        help="CSV path for step-by-step (long format) logging",
    )
    parser.add_argument(
        "--hash_method", dest="hash_method",
        default="phash", type=str,
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

    # ---- Step-by-step CSV logger (long format) — identical schema to nhash ----
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
                f"source={args.source}; lr={args.learning_rate}; "
                f"opt={args.optimizer}; ssim_w={args.ssim_weight}; "
                f"edges_only={args.edges_only}; hamming_T={args.hamming}; "
                f"check_interval={args.check_interval}; "
                f"threads={args.num_threads}; temperature={args.temperature}"
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        list(executor.map(thread_function, range(args.num_threads)))

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
