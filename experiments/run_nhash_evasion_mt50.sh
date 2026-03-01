#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG=./logs/attack_steps_nhash_evasion_mt50.csv
rm -f "$LOG"

PYTHONPATH=. python fdeph_eval/attacks/nhash_evasion_steps.py \
  --source ./inputs \
  --output_folder ./evasion_attack_outputs \
  --experiment_name nhash_evasion_mt50 \
  --threads 4 \
  --check_interval 1 \
  --learning_rate 1e-3 \
  --optimizer Adam \
  --ssim_weight 5 \
  --hamming 0.10 \
  --step_log_csv "$LOG" \
  --hash_method nhash \
  --attack_type evasion