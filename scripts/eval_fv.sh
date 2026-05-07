#!/bin/bash
# Evaluate FV-based task execution (Section A.6).
# Required env vars:
#   MODEL    - key in MODEL_NAME_DICT
# Optional env vars:
#   M        - fraction of heads to use as the FV (default: 0.02)
#   SEED, FORCE, OUT
set -e

MODEL=${MODEL:-160m}
M=${M:-0.02}
SEED=${SEED:-42}
FORCE=${FORCE:-0}
OUT=${OUT:-./outputs}

# FV from top-scored heads
python src/evaluate_function_vector.py --model_name "$MODEL" --num_fv_heads "$M" \
    --randomize 0 --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
# Random-head baseline
python src/evaluate_function_vector.py --model_name "$MODEL" --num_fv_heads "$M" \
    --randomize 1 --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
