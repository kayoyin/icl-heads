#!/bin/bash
# Compute FV scores for every attention head in $MODEL via causal mediation.
# Optional env vars:
#   MODEL  - key in MODEL_NAME_DICT (default: 160m)
#   CKPT   - Pythia checkpoint step (optional)
#   SEED   - random seed (default: 42)
#   FORCE  - 1 to overwrite existing output (default: 0)
#   OUT    - save path root (default: ./outputs)
set -e

MODEL=${MODEL:-160m}
SEED=${SEED:-42}
FORCE=${FORCE:-0}
OUT=${OUT:-./outputs}

if [ -z "$CKPT" ]; then
    python src/find_fv_heads.py --model_name "$MODEL" --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
else
    python src/find_fv_heads.py --model_name "$MODEL" --ckpt "$CKPT" --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
fi
