"""
Author: Avijit Roy
Project: FDEPH - Security Evaluation of Perceptual Image Hashing

Derived from:
  Learning-to-Break-Deep-Perceptual-Hashing (FAccT 2022)
  File: adv2_evasion_attack.py

Modifications:
  - Added step/time logging (long-format CSV) for attack efficiency analysis
  - No changes to the original repository scripts
"""
import argparse
import os
from os.path import isfile, join
from random import randint

import torch
import torchvision.transforms as T
from onnx import load_model
from skimage import feature
from skimage.color import rgb2gray
from tqdm import tqdm

from models.neuralhash import NeuralHash
from losses.mse_loss import mse_loss
from losses.quality_losses import ssim_loss
from utils.hashing import compute_hash, load_hash_matrix
from utils.image_processing import load_and_preprocess_img, save_images
from fdeph_eval.utils.structured_logger import StructuredCSVLogger
# from utils.logger import Logger
from metrics.hamming_distance import hamming_distance
import threading
import concurrent.futures
from itertools import repeat
import copy
import time

images_lock = threading.Lock()


# def optimization_thread(url_list, device, seed, loss_fkt, logger, args):
def optimization_thread(url_list, device, seed, loss_fkt, step_logger, args):
    # Store and reload source image to avoid image changes due to different formats
    id = randint(1, 10000000)
    temp_img = f'curr_image_{id}'
    model = NeuralHash()
    model.load_state_dict(torch.load('./models/model.pth', weights_only=True))
    model.to(device)
    while True:
        with images_lock:
            if not url_list:
                break
            img = url_list.pop(0)
        print('Thread working on ' + img)
        if args.optimize_original:
            resize = T.Resize((360, 360))
            source = load_and_preprocess_img(img, device, resize=False)
        else:
            source = load_and_preprocess_img(img, device, resize=True)
        input_file_name = img.rsplit(sep='/', maxsplit=1)[1].split('.')[0]
        if args.output_folder != '':
            save_images(source, args.output_folder, f'{input_file_name}')
        orig_image = source.clone()
        # Compute original hash
        with torch.no_grad():
            if args.optimize_original:
                outputs_unmodified = model(resize(source))
            else:
                outputs_unmodified = model(source)
            unmodified_hash_bin = compute_hash(
                outputs_unmodified, seed, binary=True)
            unmodified_hash_hex = compute_hash(
                outputs_unmodified, seed, binary=False)

        # Compute edge mask
        if args.edges_only:
            transform = T.Compose(
                [T.ToPILImage(), T.Grayscale(), T.ToTensor()])
            image_gray = transform(source.squeeze()).squeeze()
            image_gray = image_gray.cpu().numpy()
            edges = feature.canny(image_gray, sigma=3).astype(int)
            edge_mask = torch.from_numpy(edges).to(device)

        # Set up optimizer
        source.requires_grad = True
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                params=[source], lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params=[source], lr=args.learning_rate)
        else:
            raise RuntimeError(
                f'{args.optimizer} is no valid optimizer class. Please select --optimizer out of [Adam, SGD]')

        # Optimization cycle
        print(f'\nStart optimizing on {img}')
        for i in range(10000):
            with torch.no_grad():
                source.data = torch.clamp(source, min=-1, max=1)
            if args.optimize_original:
                outputs_source = model(resize(source))
            else:
                outputs_source = model(source)
            target_loss = - \
                loss_fkt(outputs_source, unmodified_hash_bin, seed)
            visual_loss = -ssim_loss(orig_image, source)
            optimizer.zero_grad()
            total_loss = target_loss + 0.99**i * args.ssim_weight * visual_loss
            total_loss.backward()
            if args.edges_only:
                optimizer.param_groups[0]['params'][0].grad *= edge_mask
            optimizer.step()

            # # Check for hash changes
            # if i % args.check_interval == 0:
            #     with torch.no_grad():
            #         save_images(source, './temp', temp_img)
            #         current_img = load_and_preprocess_img(
            #             f'./temp/{temp_img}.png', device, resize=True)
            #         check_output = model(current_img)
            #         source_hash_hex = compute_hash(check_output, seed)
            #         source_hash_bin = compute_hash(
            #             check_output, seed, binary=True)

            #         # Log results and finish if hash has changed
            #         if source_hash_hex != unmodified_hash_hex:
            #             if hamming_distance(source_hash_bin.unsqueeze(0), unmodified_hash_bin.unsqueeze(0)) >= args.hamming:
            #                 optimized_file = f'{args.output_folder}/{input_file_name}_opt'
            #                 if args.output_folder != '':
            #                     save_images(source, args.output_folder,
            #                                 f'{input_file_name}_opt')
            #                 # Compute metrics in the [0, 1] space
            #                 l2_distance = torch.norm(
            #                     ((current_img + 1) / 2) - ((orig_image + 1) / 2), p=2)
            #                 linf_distance = torch.norm(
            #                     ((current_img + 1) / 2) - ((orig_image + 1) / 2), p=float("inf"))
            #                 ssim_distance = ssim_loss(
            #                     (current_img + 1) / 2, (orig_image + 1) / 2)
            #                 print(
            #                     f'Finishing after {i+1} steps - L2 distance: {l2_distance:.4f} - L-Inf distance: {linf_distance:.4f} - SSIM: {ssim_distance:.4f}')

            #                 logger_data = [img, optimized_file + '.png', l2_distance.item(),
            #                                linf_distance.item(), ssim_distance.item(), i+1, target_loss.item()]
            #                 logger.add_line(logger_data)
            #                 break
            
            # Check & log at interval AR
            if i % args.check_interval == 0:
                with torch.no_grad():
                    # Timing
                    if i == 0:
                        attack_start = time.perf_counter()
                    elapsed_ms = (time.perf_counter() - attack_start) * 1000.0

                    # Re-save & reload to match original behavior
                    save_images(source, './temp', temp_img)
                    current_img = load_and_preprocess_img(
                        f'./temp/{temp_img}.png', device, resize=True)

                    check_output = model(current_img)
                    source_hash_hex = compute_hash(check_output, seed)
                    source_hash_bin = compute_hash(check_output, seed, binary=True)

                    # Distances (raw + normalized)
                    dist_raw = hamming_distance(
                        source_hash_bin.unsqueeze(0),
                        unmodified_hash_bin.unsqueeze(0),
                        normalize=False
                    )
                    dist_norm = hamming_distance(
                        source_hash_bin.unsqueeze(0),
                        unmodified_hash_bin.unsqueeze(0),
                        normalize=True
                    )

                    # Distortion metrics in [0,1] space (same as original)
                    l2_distance = torch.norm(
                        ((current_img + 1) / 2) - ((orig_image + 1) / 2), p=2
                    )
                    linf_distance = torch.norm(
                        ((current_img + 1) / 2) - ((orig_image + 1) / 2), p=float("inf")
                    )
                    ssim_distance = ssim_loss(
                        (current_img + 1) / 2, (orig_image + 1) / 2
                    )

                    # Success condition (same meaning as original script):
                    # - hash changed AND Hamming distance >= threshold
                    success = int(
                        (source_hash_hex != unmodified_hash_hex) and (dist_norm >= args.hamming)
                    )

                    # Write one step row (long format)
                    step_logger.log_row({
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
                        "source_path": img
                    })

                    # If successful, save output and stop
                    if success == 1:
                        if args.output_folder != '':
                            save_images(source, args.output_folder, f'{input_file_name}_opt')
                        print(
                            f'Finishing after {i+1} steps - '
                            f"L2: {float(l2_distance.item() if hasattr(l2_distance, 'item') else l2_distance):.4f} - "
                            f"LInf: {float(linf_distance.item() if hasattr(linf_distance, 'item') else linf_distance):.4f} - "
                            f"SSIM: {float(ssim_distance.item() if hasattr(ssim_distance, 'item') else ssim_distance):.4f} - "
                            f"d_raw: {float(dist_raw.item() if hasattr(dist_raw, 'item') else dist_raw):.4f} - "
                            f"d_norm: {float(dist_norm.item() if hasattr(dist_norm, 'item') else dist_norm):.4f}"
                        )
                        break            #--AR
    # os.remove(f'./temp/{temp_img}.png')
    temp_path = f'./temp/{temp_img}.png'
    if os.path.exists(temp_path):
        os.remove(temp_path)    


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='inputs/source.png', help='image to manipulate')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-3,
                        type=float, help='step size of PGD optimization step')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam',
                        type=str, help='kind of optimizer')
    parser.add_argument('--ssim_weight', dest='ssim_weight', default=5,
                        type=float, help='weight of ssim loss')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='change_hash_attack', type=str, help='name of the experiment and logging file')
    parser.add_argument('--output_folder', dest='output_folder',
                        default='evasion_attack_outputs', type=str, help='folder to save optimized images in')
    parser.add_argument('--edges_only', dest='edges_only',
                        action='store_true', help='Change only pixels of edges')
    parser.add_argument('--optimize_original', dest='optimize_original',
                        action='store_true', help='Optimize resized image')
    parser.add_argument('--sample_limit', dest='sample_limit',
                        default=10000000, type=int, help='Maximum of images to be processed')
    parser.add_argument('--hamming', dest='hamming',
                        default=0.00001, type=float, help='Minimum Hamming distance to stop')
    parser.add_argument('--threads', dest='num_threads',
                        default=4, type=int, help='Number of parallel threads')
    parser.add_argument('--check_interval', dest='check_interval',
                        default=1, type=int, help='Hash change interval checking')
    # ---- FDEPH Eval Logging ----
    parser.add_argument('--step_log_csv', dest='step_log_csv',
                        default='./logs/attack_steps_nhash_evasion.csv', type=str,
                        help='CSV path for step-by-step (long format) logging')
    parser.add_argument('--hash_method', dest='hash_method',
                        default='nhash', type=str, help='hash method label for logs')
    parser.add_argument('--attack_type', dest='attack_type',
                        default='evasion', type=str, help='attack type label for logs')
    args = parser.parse_args()

    # Create temp folder
    os.makedirs('./temp', exist_ok=True)

    # Load and prepare components
    start = time.time()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = load_hash_matrix()
    seed = torch.tensor(seed).to(device)

    # Prepare output folder
    if args.output_folder != '':
        try:
            os.mkdir(args.output_folder)
        except:
            if not os.listdir(args.output_folder):
                print(
                    f'Folder {args.output_folder} already exists and is empty.')
            else:
                print(
                    f'Folder {args.output_folder} already exists and is not empty.')

    # # Prepare logging
    # logging_header = ['file', 'optimized_file', 'l2',
    #                   'l_inf', 'ssim', 'steps', 'target_loss', 'Hamming']
    # logger = Logger(args.experiment_name, logging_header, output_dir='./logs')
    # logger.add_line(['Hyperparameter', args.source, args.learning_rate,
    #                  args.optimizer, args.ssim_weight, args.edges_only, args.hamming])
    
    # ---- Step-by-step CSV logger (long format) AR ----
    step_header = [
        "image_id", "hash_method", "attack_type",
        "step", "elapsed_ms",
        "dist_raw", "dist_norm",
        "l2", "linf", "ssim",
        "success",
        "source_path"
    ]
    step_logger = StructuredCSVLogger(args.step_log_csv, step_header)

    # Log a single "hyperparams" row as step=0 for reproducibility
    step_logger.log_row({
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
        "source_path": f"source={args.source}; lr={args.learning_rate}; opt={args.optimizer}; "
                       f"ssim_w={args.ssim_weight}; edges_only={args.edges_only}; "
                       f"hamming_T={args.hamming}; check_interval={args.check_interval}; "
                       f"threads={args.num_threads}"
    })   

    # define loss function
    loss_function = mse_loss

    # Load images
    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        images = [join(args.source, f) for f in os.listdir(
            args.source) if isfile(join(args.source, f))]
        images = sorted(images)
    else:
        raise RuntimeError(f'{args.source} is neither a file nor a directory.')
    images = images[:args.sample_limit]

    # Start threads
    # def thread_function(x): return optimization_thread(
    #     images, device, seed, loss_function, logger, args)
    def thread_function(_):
        return optimization_thread(images, device, seed, loss_function, step_logger, args)

    # Launch exactly args.num_threads workers; each worker pops from the shared images list
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        list(executor.map(thread_function, range(args.num_threads)))

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
