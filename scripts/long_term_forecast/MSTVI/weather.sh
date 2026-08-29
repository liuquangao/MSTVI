#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

for pred_len in 96 192 336 720; do
    .venv/bin/python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "weather_96_${pred_len}" \
        --model MSTVI \
        --data custom \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --features M \
        --freq t \
        --seq_len 96 \
        --label_len 48 \
        --pred_len "$pred_len" \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n1 512 \
        --n2 256 \
        --n3 128 \
        --batch_size 32 \
        --train_epochs 30 \
        --patience 5 \
        --learning_rate 0.0001 \
        --num_workers 4 \
        --itr 1
done
