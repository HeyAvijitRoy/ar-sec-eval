# experiments/run_collision_sweep.py
# Author: Avijit Roy — FDEPH Project
#
# Phase 3: Targeted hash collision sweep on both NeuralHash and pHash.
#
# Step 1: Check that hash CSVs exist (print instructions if missing).
# Step 2: Run NeuralHash collision attack on inputs_500 → target train set.
# Step 3: Run pHash collision attack on inputs_500 → target train set.
#
# Separate output folders are used per hash type to avoid cross-contamination
# (lesson from Phase 2: mixed folders break per-method analysis).
#
# NOTE on ssim_weight:
#   NeuralHash collision uses ssim_weight=100 (original Struppek convention).
#   pHash collision uses ssim_weight=5 (matches pHash evasion convention).
#   Using 100 for pHash causes the regularization to dominate the BCE collision
#   signal entirely, resulting in 0 successful collisions.
#
# Usage:
#   PYTHONPATH=. python experiments/run_collision_sweep.py [--threads N]

import argparse
import datetime
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ---- Attack scripts ---------------------------------------------------------
NHASH_ATTACK_SCRIPT = os.path.join(REPO_ROOT, 'adv1_collision_attack.py')
PHASH_ATTACK_SCRIPT = os.path.join(
    REPO_ROOT, 'fdeph_eval', 'attacks', 'adv1_collision_attack_phash.py')

# ---- Source images ----------------------------------------------------------
INPUT_DIR = os.path.join(REPO_ROOT, 'inputs', 'inputs_500')

# ---- Target hash CSVs (must be generated before this sweep) ----------------
NHASH_CSV = os.path.join(
    REPO_ROOT, 'dataset_hashes', 'imagenette_train_nhash_hashes.csv')
PHASH_CSV = os.path.join(
    REPO_ROOT, 'dataset_hashes', 'imagenette_train_phash_hashes.csv')

# ---- Target image folder ---------------------------------------------------
TARGET_IMAGE_FOLDER = os.path.join(REPO_ROOT, 'data', 'imagenette2-320', 'train')

# ---- Output folders (separate per hash type) --------------------------------
NHASH_OUT_DIR = os.path.join(REPO_ROOT, 'collision_attack_outputs_nhash')
PHASH_OUT_DIR = os.path.join(REPO_ROOT, 'collision_attack_outputs_phash')

# ---- Shared hyperparameters ------------------------------------------------
LR          = '1e-3'
OPT         = 'Adam'
CHECK_INT   = '10'

# ssim_weight is intentionally different per attack — see NOTE above
NHASH_SSIM_W = '100'   # NeuralHash: original Struppek convention
PHASH_SSIM_W = '5'     # pHash: matches evasion convention; 100 kills convergence


def check_hash_csvs():
    """Return True if both hash CSVs exist, print generation instructions if not."""
    missing = []
    if not os.path.isfile(NHASH_CSV):
        missing.append(NHASH_CSV)
    if not os.path.isfile(PHASH_CSV):
        missing.append(PHASH_CSV)

    if not missing:
        return True

    print('\n' + '=' * 60)
    print('  ERROR: Required hash CSV(s) not found.')
    print('=' * 60)
    for path in missing:
        print(f'  Missing: {path}')

    print('\n  Generate them first with:')
    if not os.path.isfile(NHASH_CSV):
        print(
            '\n  # NeuralHash hashes for train set:\n'
            '  PYTHONPATH=. python utils/compute_dataset_hashes.py \\\n'
            '      --source data/imagenette2-320/train \\\n'
            '      --output_path dataset_hashes/imagenette_train_nhash_hashes.csv'
        )
    if not os.path.isfile(PHASH_CSV):
        print(
            '\n  # pHash hashes for train set:\n'
            '  PYTHONPATH=. python utils/compute_dataset_hashes_phash.py \\\n'
            '      --source data/imagenette2-320/train \\\n'
            '      --output_path dataset_hashes/imagenette_train_phash_hashes.csv'
        )
    print()
    return False


def run_attack(label, script, extra_args, num_threads):
    """Run a single collision attack subprocess and return True on success."""
    cmd = [
        sys.executable, script,
        '--source',         INPUT_DIR,
        '--learning_rate',  LR,
        '--optimizer',      OPT,
        '--check_interval', CHECK_INT,
        '--threads',        str(num_threads),
    ] + extra_args

    env = os.environ.copy()
    env['PYTHONPATH'] = REPO_ROOT

    t_start = datetime.datetime.now()
    print(f'\n{"=" * 60}')
    print(f'  Phase 3 — {label}')
    print(f'  Started : {t_start.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Threads : {num_threads}')
    print(f'{"=" * 60}')

    result = subprocess.run(cmd, env=env)

    t_end = datetime.datetime.now()
    elapsed = t_end - t_start
    ok = (result.returncode == 0)

    print(f'\n  Finished : {t_end.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Elapsed  : {str(elapsed).split(".")[0]}')
    print(f'  Status   : {"OK" if ok else f"FAILED (rc={result.returncode})"}')

    return ok


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3 collision sweep: NeuralHash + pHash.')
    parser.add_argument('--threads', dest='num_threads', type=int, default=8,
                        help='Parallel worker threads per attack (default: 8)')
    args = parser.parse_args()

    # Step 1 — Gate on hash CSV existence
    if not check_hash_csvs():
        sys.exit(1)

    os.makedirs(NHASH_OUT_DIR, exist_ok=True)
    os.makedirs(PHASH_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, 'logs'), exist_ok=True)

    results = {}

    # Step 2 — NeuralHash collision attack (ssim_weight=100)
    results['nhash'] = run_attack(
        label='NeuralHash Collision Attack',
        script=NHASH_ATTACK_SCRIPT,
        extra_args=[
            '--ssim_weight',        NHASH_SSIM_W,
            '--target_hashset',     NHASH_CSV,
            '--target_image_folder', TARGET_IMAGE_FOLDER,
            '--output_folder',      NHASH_OUT_DIR,
            '--experiment_name',    'nhash_collision_imagenette500',
        ],
        num_threads=args.num_threads,
    )

    # Step 3 — pHash collision attack (ssim_weight=5, not 100)
    results['phash'] = run_attack(
        label='pHash Collision Attack',
        script=PHASH_ATTACK_SCRIPT,
        extra_args=[
            '--ssim_weight',        PHASH_SSIM_W,
            '--target_hashset',     PHASH_CSV,
            '--target_image_folder', TARGET_IMAGE_FOLDER,
            '--output_folder',      PHASH_OUT_DIR,
            '--experiment_name',    'phash_collision_imagenette500',
        ],
        num_threads=args.num_threads,
    )

    # ---- Summary ------------------------------------------------------------
    print(f'\n{"=" * 60}')
    print('  PHASE 3 COLLISION SWEEP SUMMARY')
    print(f'{"=" * 60}')
    all_ok = True
    for key, ok in results.items():
        status = 'COMPLETED' if ok else 'FAILED'
        print(f'  {key.upper():<10}  {status}')
        if not ok:
            all_ok = False

    print(f'{"=" * 60}')
    if all_ok:
        print('  All collision attacks completed successfully.')
        print(f'  NeuralHash outputs : {NHASH_OUT_DIR}')
        print(f'  pHash outputs      : {PHASH_OUT_DIR}')
        print(f'  Logs               : {os.path.join(REPO_ROOT, "logs")}')
    else:
        print('  WARNING: one or more attacks failed — check output above.')
    print()


if __name__ == '__main__':
    main()