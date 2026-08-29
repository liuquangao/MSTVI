#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

for pred_len in 96 192 336 720; do
    .venv/bin/python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "traffic_96_${pred_len}" \
        --model MSTVI \
        --data custom \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --features M \
        --freq h \
        --seq_len 96 \
        --label_len 48 \
        --pred_len "$pred_len" \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n1 512 \
        --n2 256 \
        --n3 128 \
        --batch_size 32 \
        --train_epochs 30 \
        --patience 5 \
        --learning_rate 0.001 \
        --num_workers 4 \
        --itr 1
done
