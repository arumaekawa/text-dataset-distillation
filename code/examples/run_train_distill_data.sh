#!/bin/bash -e

./run.sh --mode distill --accum_loss --drop_p 0 "$@"
