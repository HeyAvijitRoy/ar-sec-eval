"""
Isolated PDQ white-box evasion attack driven by a differentiable surrogate.

The optimizer follows the surrogate gradients, but the only success criterion is
the exact PDQ hard hash from ``pdqhash``.
"""

import argparse
import concurrent.futures
import os
import threading
import time
from os.path import isfile, join
from random import randint

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage import feature

from fdeph_eval.attacks.pdq.pdq_surrogate import compute_pdq_soft_surrogate
from fdeph_eval.utils.structured_logger import StructuredCSVLogger
from losses.quality_losses import ssim_loss
from metrics.hamming_distance import hamming_distance
from utils.image_processing import load_and_preprocess_img, save_images
from utils.pdq_torch import compute_pdq_hard

images_lock = threading.Lock()


def surrogate_flip_loss(soft_hash: torch.Tensor, surrogate_orig_hash: torch.Tensor) -> torch.Tensor:
    flipped_target = 1.0 - surrogate_orig_hash.float()
    soft_hash = soft_hash.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(soft_hash, flipped_target)


def optimization_thread(url_list, device, step_logger, args):
    thread_id = randint(1, 10000000)
    temp_img = f"curr_pdq_wb_{thread_id}"

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
            official_orig_hash, official_orig_hex, _ = compute_pdq_hard(source)
            _, surrogate_orig_hash, surrogate_median = compute_pdq_soft_surrogate(source)

        edge_mask = None
        if args.edges_only:
            transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])
            image_gray = transform(source.squeeze()).squeeze().cpu().numpy()
            edges = feature.canny(image_gray, sigma=3).astype(int)
            edge_mask = torch.from_numpy(edges).to(device)

        source.requires_grad = True
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(params=[source], lr=args.learning_rate)
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(params=[source], lr=args.learning_rate)
        else:
            raise RuntimeError(f"Unsupported optimizer: {args.optimizer}")

        print(f"\nStart optimizing on {img}")
        attack_start = time.perf_counter()
        best_dist_norm = 0.0

        for i in range(args.max_steps):
            with torch.no_grad():
                source.data = source.data.clamp(-1, 1)

            soft_hash, _, _ = compute_pdq_soft_surrogate(
                source,
                median_ref=surrogate_median,
                temperature=args.temperature,
            )
            target_loss = surrogate_flip_loss(soft_hash, surrogate_orig_hash)
            visual_loss = -ssim_loss(orig_image, source)
            total_loss = target_loss + (0.99 ** i) * args.ssim_weight * visual_loss

            optimizer.zero_grad()
            total_loss.backward()

            if args.edges_only and edge_mask is not None:
                optimizer.param_groups[0]["params"][0].grad *= edge_mask

            optimizer.step()

            if i % args.check_interval == 0:
                with torch.no_grad():
                    elapsed_ms = (time.perf_counter() - attack_start) * 1000.0

                    save_images(source, "./temp", temp_img)
                    current_img = load_and_preprocess_img(
                        f"./temp/{temp_img}.png", device, resize=True
                    )
                    current_hash_bin, current_hash_hex, _ = compute_pdq_hard(current_img)

                    dist_raw = hamming_distance(
                        current_hash_bin.unsqueeze(0),
                        official_orig_hash.unsqueeze(0),
                        normalize=False,
                    )
                    dist_norm = hamming_distance(
                        current_hash_bin.unsqueeze(0),
                        official_orig_hash.unsqueeze(0),
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
                        (current_hash_hex != official_orig_hex)
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
    parser = argparse.ArgumentParser(description="PDQ white-box surrogate evasion attack.")
    parser.add_argument("--source", type=str, default="./inputs/pdq_subset_10")
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--ssim_weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--output_folder", type=str, default="./evasion_attack_outputs_pdq_whitebox")
    parser.add_argument("--edges_only", action="store_true")
    parser.add_argument("--sample_limit", type=int, default=10)
    parser.add_argument("--hamming", type=float, default=0.08)
    parser.add_argument("--threads", dest="num_threads", type=int, default=2)
    parser.add_argument("--check_interval", type=int, default=1)
    parser.add_argument(
        "--step_log_csv",
        type=str,
        default="./logs/attack_steps_pdq_whitebox_subset10.csv",
    )
    parser.add_argument("--hash_method", type=str, default="pdq")
    parser.add_argument("--attack_type", type=str, default="evasion_whitebox_pdq")
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
                f"source={args.source}; lr={args.learning_rate}; opt={args.optimizer}; "
                f"ssim_w={args.ssim_weight}; temperature={args.temperature}; "
                f"max_steps={args.max_steps}; hamming_T={args.hamming}; "
                f"threads={args.num_threads}"
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
