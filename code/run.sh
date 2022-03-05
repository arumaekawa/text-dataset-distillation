#!/bin/bash -e

source ./env

python main.py \
    --model_root_dir $MODEL_ROOT_DIR \
    --data_root_dir $DATA_ROOT_DIR \
    "$@"
