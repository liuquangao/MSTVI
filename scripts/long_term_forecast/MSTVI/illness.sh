#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

for pred_len in 24 36 48 60; do
    .venv/bin/python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "illness_36_${pred_len}" \
        --model MSTVI \
        --data custom \
        --root_path ./dataset/illness/ \
        --data_path national_illness.csv \
        --features M \
        --freq w \
        --seq_len 36 \
        --label_len 18 \
        --pred_len "$pred_len" \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n1 512 \
        --n2 256 \
        --n3 128 \
        --batch_size 16 \
        --train_epochs 10 \
        --patience 3 \
        --learning_rate 0.0001 \
        --num_workers 4 \
        --itr 1
done
