#!/usr/bin/env bash
set -e

PROMPT="A charming little cat, adorned with a delicate red bow tie, strolls daintily across the emerald green grass."
NEGATIVE_PROMPT=""
CACHE_DIR="/data/pretrained_models/"
OUTPUT_DIR="/home/hyp/results/wan"

HEIGHT=768
WIDTH=1280
FRAMES=69
TOPK=80
# HEIGHT=384
# WIDTH=512
# FRAMES=49
# TOPK=60
MAX_SEQ=512
SPARSE_TYPE='mod-dit'

nohup python wan_test.py \
  --prompt "$PROMPT" \
  --negative_prompt "$NEGATIVE_PROMPT" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --height $HEIGHT \
  --width $WIDTH \
  --frames $FRAMES \
  --top_k $TOPK \
  --max_seq_length $MAX_SEQ \
  --sparse_type $SPARSE_TYPE \
  --warmup_steps 10 \
  --predict_T 15 \