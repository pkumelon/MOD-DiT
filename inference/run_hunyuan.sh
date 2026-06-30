#!/usr/bin/env bash
set -e

# PROMPT="A charming little cat, adorned with a delicate red bow tie, strolls daintily across the emerald green grass."
PROMPT="In a sunlit garden, a sleek black cat with piercing green eyes sits poised on a wooden fence, its tail flicking with curiosity. Nearby, a vibrant blue jay perches on a blooming cherry blossom branch, its feathers shimmering in the sunlight. The cat's gaze is fixed on the bird, but there's a sense of peaceful coexistence rather than predation. The bird chirps melodiously, and the cat's ears twitch in response, creating a harmonious scene. The garden, filled with colorful flowers and lush greenery, serves as a tranquil backdrop to this delicate interaction between the two creatures."
NEGATIVE_PROMPT=""
CACHE_DIR="/data/pretrained_models/"
OUTPUT_DIR="/home/hyp/results/"

HEIGHT=768
WIDTH=1280
FRAMES=117
TOPK=300
MAX_SEQ=256
SPARSE_TYPE='mod-dit'

nohup python hunyuan_test.py \
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