#!/usr/bin/env bash
set -e

PROMPT="A dog wearing sunglasses, running in a park."
CACHE_DIR="/data/pretrained_models/CogVideoX"
OUTPUT_DIR="/home/hyp/results"

HEIGHT=384
WIDTH=512
FRAMES=49
TOPK=50
MAX_SEQ=128
SPARSE_TYPE=''

nohup python cogvideo_test.py \
  --prompt "$PROMPT" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --height $HEIGHT \
  --width $WIDTH \
  --frames $FRAMES \
  --top_k $TOPK \
  --max_seq_length $MAX_SEQ \
  --sparse_type $SPARSE_TYPE \
  --warmup_steps 11 \