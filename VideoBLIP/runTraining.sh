#!/bin/bash

python -m scripts.DVC.train_dvc \
    --model_name_or_path kpyu/video-blip-opt-2.7b-ego4d \
    --num_subsample_frames 30 \
    --train_narrated_actions_dir Data \
    --val_narrated_actions_dir Data \
    --output_dir Output \
    --num_train_epochs 5 \
    --warmup_steps 1000 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.05 \
    --dataloader_num_workers 2 \
    --bf16 False \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10
