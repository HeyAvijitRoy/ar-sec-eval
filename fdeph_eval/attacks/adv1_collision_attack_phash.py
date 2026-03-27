# fdeph_eval/attacks/adv1_collision_attack_phash.py
# Author: Avijit Roy — FDEPH Project
#
# Targeted hash collision attack on pHash.
# Mirror of adv1_collision_attack.py (NeuralHash collision) but targets pHash.
#
# Derived from:
#   adv1_collision_attack.py          (NeuralHash collision, Struppek et al.)
#   fdeph_eval/attacks/phash_evasion_steps.py  (pHash evasion reference)
#
# KEY DIFFERENCE FROM EVASION:
#   Evasion maximises distance FROM source hash (push bits away from original).
#   Collision minimises distance TO target hash (push bits toward target).
#
#   Loss direction:
#     target_loss = mean(soft_hash * (1 - 2 * target_hash_bin))
#     When target_bit=1: multiplier=-1 → minimising pushes soft_hash UP  → bit=1 ✓
#     When target_bit=0: multiplier=+1 → minimising lowers soft_hash DOWN → bit=0 ✓
#
#   median_ref is frozen to the SOURCE image's own DCT median (same as evasion).
#   Using the target image's median caused sigmoid saturation because source and
#   target DCT coefficient distributions are unrelated — gradients went to zero.
#   The source median keeps the sigmoid in its active region throughout training.
#   The target hash tensor already encodes where each bit needs to go.

import argparse
import concurrent.futures
import os
import csv
import threading
from os.path import isfile, join
from random import randint

import torch
import torch.nn.functional as F

from losses.quality_losses import ssim_loss
from utils.image_processing import load_and_preprocess_img, save_images
from utils.logger import Logger
from utils.phash_torch import compute_phash_hard, compute_phash_soft

images_lock = threading.Lock()


def optimization_thread(url_list, device, logger, args,
                        target_hashes, bin_hex_hash_dict, hex_path_dict):
    print(f'Process {threading.get_ident()} started')

    id_ = randint(1, 1000000)
    temp_img = f'curr_image_{id_}'

    while True:
        # Thread-safe pop from shared image list
        with images_lock:
            if not url_list:
                break
            img = url_list.pop(0)

        print(f'Thread working on {img}')

        # Load source image (round-trip save/reload to avoid format artefacts)
        source = load_and_preprocess_img(img, device, resize=True)
        input_file_name = img.rsplit(sep='/', maxsplit=1)[1].split('.')[0]

        if args.output_folder != '':
            save_images(source, args.output_folder, f'{input_file_name}')

        source_orig = source.clone()

        # ---- Find nearest target hash (minimum Hamming distance) ------------
        with torch.no_grad():
            source_hard, _, _ = compute_phash_hard(source)
            hamming_dist = torch.norm(
                source_hard.float() - target_hashes.float(), p=1, dim=1
            ) / source_hard.shape[0]
            _, idx = torch.min(hamming_dist, dim=0)
            target_hash = target_hashes[idx.item()]   # [64] int tensor
            target_hash_str = ''.join(
                ['1' if b > 0.5 else '0' for b in target_hash.tolist()]
            )
            target_hash_hex = bin_hex_hash_dict[target_hash_str]
            target_image_path = hex_path_dict[target_hash_hex]

        # ---- Load target image (for saving only — NOT for median) -----------
        # We load this once here so it's available when we save on success.
        with torch.no_grad():
            target_img_tensor = load_and_preprocess_img(
                target_image_path, device, resize=True)

        # ---- Freeze median_ref to SOURCE image's own DCT median -------------
        # IMPORTANT: do NOT use the target image's median here.
        # The target median causes sigmoid saturation because source and target
        # DCT distributions are unrelated — this kills the gradient entirely.
        # The source median keeps sigmoid(coeff - median)*τ in its active
        # region throughout optimization. The target hash tensor handles
        # the direction (which bits to push up vs down).
        with torch.no_grad():
            _, _, source_median = compute_phash_hard(source)

        # ---- Optimizer setup ------------------------------------------------
        source.requires_grad = True
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                params=[source], lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                params=[source], lr=args.learning_rate)
        else:
            raise RuntimeError(
                f'{args.optimizer} is not a valid optimizer. '
                'Choose from [Adam, SGD].')

        print(f'\nStart collision optimisation on {img}')
        print(f'  Target hash : {target_hash_hex}')
        print(f'  Target image: {target_image_path}')

        # ---- Optimisation loop ----------------------------------------------
        for i in range(10000):
            with torch.no_grad():
                source.data = torch.clamp(source.data, min=-1, max=1)

            # Differentiable pHash with SOURCE frozen median for stable gradients
            soft_hash, _, _ = compute_phash_soft(
                source,
                median_ref=source_median,
                temperature=args.temperature,
            )

            # Collision loss: push each bit of soft_hash toward the target bit.
            #   bit=1 → (1 - 2*1) = -1 → minimising raises soft_hash → output 1 ✓
            #   bit=0 → (1 - 2*0) = +1 → minimising lowers soft_hash → output 0 ✓
            # target_loss = torch.mean(
            #     soft_hash * (1.0 - 2.0 * target_hash.float())
            # )
            import torch.nn.functional as F
            target_loss = F.binary_cross_entropy(soft_hash, target_hash.float())

            # Visual quality regularisation (same convention as NeuralHash version)
            visual_loss = -1.0 * args.ssim_weight * ssim_loss(source_orig, source)

            optimizer.zero_grad()
            total_loss = target_loss + visual_loss
            total_loss.backward()
            optimizer.step()

            # ---- Check for exact hash match at interval ---------------------
            if i % args.check_interval == 0:
                with torch.no_grad():
                    save_images(source, './temp', temp_img)
                    current_img = load_and_preprocess_img(
                        f'./temp/{temp_img}.png', device, resize=True)
                    _, source_hash_hex, _ = compute_phash_hard(current_img)

                    # Success criterion: exact hex match (Hamming distance = 0)
                    if source_hash_hex == target_hash_hex:
                        if args.output_folder != '':
                            # Save adversarial (poisoned) image
                            save_images(source, args.output_folder,
                                        f'{input_file_name}_opt')
                            # Save target image for side-by-side inspection
                            save_images(target_img_tensor, args.output_folder,
                                        f'{input_file_name}_target')

                        optimized_file = os.path.join(
                            args.output_folder, f'{input_file_name}_opt.png')
                        target_file = os.path.join(
                            args.output_folder, f'{input_file_name}_target.png')

                        # Distortion metrics in [0, 1] pixel space
                        l2_distance = torch.norm(
                            ((current_img + 1) / 2) - ((source_orig + 1) / 2), p=2)
                        linf_distance = torch.norm(
                            ((current_img + 1) / 2) - ((source_orig + 1) / 2),
                            p=float('inf'))
                        ssim_distance = ssim_loss(
                            (current_img + 1) / 2, (source_orig + 1) / 2)

                        print(
                            f'Finishing after {i+1} steps - '
                            f'L2: {l2_distance:.4f} - '
                            f'L-Inf: {linf_distance:.4f} - '
                            f'SSIM: {ssim_distance:.4f}')

                        logger_data = [
                            img,
                            optimized_file,
                            target_file,
                            target_image_path,
                            l2_distance.item(),
                            linf_distance.item(),
                            ssim_distance.item(),
                            i + 1,
                        ]
                        logger.add_line(logger_data)
                        break

    # Clean up temp file for this thread
    temp_path = f'./temp/{temp_img}.png'
    if os.path.exists(temp_path):
        os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(
        description='Perform pHash targeted collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='inputs/source.png',
                        help='Image or directory of images to manipulate')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-3,
                        type=float, help='Step size for Adam optimisation')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam',
                        type=str, help='Optimizer class [Adam | SGD]')
    parser.add_argument('--ssim_weight', dest='ssim_weight', default=5,
                        type=float, help='Weight of SSIM visual-quality loss')
    parser.add_argument('--temperature', dest='temperature', default=20.0,
                        type=float,
                        help='Sigmoid temperature for soft pHash surrogate (default: 20)')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='phash_collision_attack', type=str,
                        help='Name of the experiment and logging file')
    parser.add_argument('--output_folder', dest='output_folder',
                        default='collision_attack_outputs_phash', type=str,
                        help='Folder to save optimised images in')
    parser.add_argument('--target_hashset', dest='target_hashset',
                        type=str,
                        help='Target hashset CSV file path '
                             '(from utils/compute_dataset_hashes_phash.py)')
    parser.add_argument('--target_image_folder', dest='target_image_folder',
                        type=str, default='',
                        help='Root folder where target images are stored '
                             '(image paths come directly from the CSV)')
    parser.add_argument('--sample_limit', dest='sample_limit',
                        default=10000000, type=int,
                        help='Maximum number of images to process')
    parser.add_argument('--threads', dest='num_threads',
                        default=8, type=int,
                        help='Number of parallel worker threads')
    parser.add_argument('--check_interval', dest='check_interval',
                        default=10, type=int,
                        help='Steps between hash-change checks')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ---- Read target hashset CSV --------------------------------------------
    # Produced by utils/compute_dataset_hashes_phash.py
    # Columns: index, image, hash_bin, hash_hex
    target_hashes = []
    bin_hex_hash_dict = {}   # hash_bin_str → hash_hex_str
    hex_path_dict = {}       # hash_hex_str → image_path

    with open(args.target_hashset, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip header
        for row in reader:
            image_path = row[1]
            hash_bin_str = row[2]
            hash_hex_str = row[3]

            hash_tensor = torch.tensor(
                [int(b) for b in list(hash_bin_str)]
            ).unsqueeze(0)  # [1, 64]

            bin_hex_hash_dict[hash_bin_str] = hash_hex_str
            hex_path_dict[hash_hex_str] = image_path
            target_hashes.append(hash_tensor)

    target_hashes = torch.cat(target_hashes, dim=0).to(device)  # [N, 64]

    # ---- Prepare output folder ----------------------------------------------
    if args.output_folder != '':
        try:
            os.mkdir(args.output_folder)
        except FileExistsError:
            if not os.listdir(args.output_folder):
                print(f'Folder {args.output_folder} already exists and is empty.')
            else:
                print(f'Folder {args.output_folder} already exists and is not empty.')

    # ---- Logging setup ------------------------------------------------------
    logging_header = ['file', 'optimized_file', 'target_file',
                      'target_image_path', 'l2', 'l_inf', 'ssim', 'steps']
    logger = Logger(args.experiment_name, logging_header, output_dir='./logs')
    logger.add_line(['Hyperparameter', args.source, args.learning_rate,
                     args.optimizer, args.ssim_weight, args.temperature])

    # ---- Load source images -------------------------------------------------
    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        images = sorted([
            join(args.source, f)
            for f in os.listdir(args.source)
            if isfile(join(args.source, f))
        ])
    else:
        raise RuntimeError(f'{args.source} is neither a file nor a directory.')
    images = images[:args.sample_limit]

    # ---- Launch worker threads ----------------------------------------------
    threads_args = (images, device, logger, args,
                    target_hashes, bin_hex_hash_dict, hex_path_dict)

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        for _ in range(args.num_threads):
            executor.submit(lambda p: optimization_thread(*p), threads_args)

    logger.finish_logging()


if __name__ == '__main__':
    os.makedirs('./temp', exist_ok=True)
    main()