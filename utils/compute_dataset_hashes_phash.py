# utils/compute_dataset_hashes_phash.py
# Author: Avijit Roy — FDEPH Project
#
# Compute pHash for every image in a folder tree and save results as CSV.
# Mirror of utils/compute_dataset_hashes.py but uses compute_phash_hard
# from utils/phash_torch.py instead of NeuralHash.
#
# Usage:
#   PYTHONPATH=. python utils/compute_dataset_hashes_phash.py \
#       --source data/imagenette2-320/train/ \
#       --output_path dataset_hashes/imagenette_train_phash_hashes.csv

import argparse
import os
import pathlib

import pandas as pd
import numpy as np

import torch
from PIL import Image
from tqdm import tqdm

from utils.phash_torch import compute_phash_hard


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute pHash for a folder of images and save to CSV.')
    parser.add_argument('--source', dest='source', type=str,
                        default='data/imagenette2-320/train',
                        help='Image file or folder to compute hashes for')
    parser.add_argument('--output_path', dest='output_path', type=str,
                        default='',
                        help='Full path for output CSV. '
                             'Default: dataset_hashes/{folder_name}_phash_hashes.csv')
    args = parser.parse_args()

    # Load images — walk recursively to cover class subfolders
    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        images = [
            os.path.join(path, name)
            for path, subdirs, files in os.walk(args.source)
            for name in files
        ]
        images = sorted(images)
    else:
        raise RuntimeError(f'{args.source} is neither a file nor a directory.')

    # Device (GPU if available)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Prepare results dataframe
    result_df = pd.DataFrame(columns=['image', 'hash_bin', 'hash_hex'])

    for img_name in tqdm(images, desc='Computing pHash'):
        # Preprocess image — same pipeline as compute_phash_hard expects
        try:
            img = Image.open(img_name).convert('RGB')
        except Exception:
            # Skip unreadable images gracefully
            continue

        img = img.resize([360, 360])
        arr = np.array(img).astype('float32') / 255.0
        arr = arr * 2.0 - 1.0
        arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])
        arr = torch.tensor(arr).to(device)

        # Compute pHash
        hard_hash, hash_hex, _ = compute_phash_hard(arr)
        hash_bin = ''.join(['1' if b > 0.5 else '0' for b in hard_hash.tolist()])

        result_df = pd.concat(
            [result_df,
             pd.DataFrame([{'image': img_name,
                            'hash_bin': hash_bin,
                            'hash_hex': hash_hex}])],
            ignore_index=True,
        )

    # Determine output path
    os.makedirs('./dataset_hashes', exist_ok=True)
    if args.output_path != '':
        out_path = args.output_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    elif os.path.isfile(args.source):
        out_path = os.path.join('./dataset_hashes',
                                f'{args.source}_phash_hashes.csv')
    else:
        folder_name = pathlib.PurePath(args.source).name
        out_path = os.path.join('./dataset_hashes',
                                f'{folder_name}_phash_hashes.csv')

    result_df.to_csv(out_path)
    print(f'Saved {len(result_df)} hashes to {out_path}')


if __name__ == '__main__':
    main()
