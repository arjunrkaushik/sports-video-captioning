#!/bin/bash

python -m scripts.DVC.test_dvc \
    --model_dir /data/kaushik3/Output_DVC \
    --test_data_dir /data/kaushik3/DVC_Data \
    --num_subsample_frames 15 \
    --num_workers 1