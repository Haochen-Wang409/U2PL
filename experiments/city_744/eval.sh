#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='eval'

mkdir -p log
mkdir -p checkpoints/result

python ../../eval.py \
    --base_size=2048 \
    --scales 1.0 \
    --config=config.yaml \
    --model_path=checkpoints/ckpt.pth \
    --save_folder=checkpoints/result/ 2>&1 | tee log/val_$now.txt
