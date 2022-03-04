#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='744_semi'

mkdir -p log

# use torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$2 \
    ../../train.py --config=config.yaml --seed 1 --port $2 2>&1 | tee log/log_$now.txt