#!/bin/bash
# Token-loss difference (ICL score from Olsson et al., 2022) under ablation.
# Required env vars:
#   MODEL          - key in MODEL_NAME_DICT
#   ABL            - fraction of heads to ablate. Use 0 for clean.
# Optional env vars:
#   CKPT           - Pythia checkpoint step
#   ABL_HEAD       - induction|fv|random (default: cycles through all three)
#   EXCL           - 1 to use ablation-with-exclusion (default: 0)
#   SEED, FORCE, OUT
set -e

MODEL=${MODEL:-160m}
ABL=${ABL:-0}
EXCL=${EXCL:-0}
SEED=${SEED:-42}
FORCE=${FORCE:-0}
OUT=${OUT:-./outputs}

CKPT_ARG=""
if [ -n "$CKPT" ]; then
    CKPT_ARG="--ckpt $CKPT"
fi

if [ "$ABL" = "0" ]; then
    python src/evaluate_icl_score.py --model_name "$MODEL" $CKPT_ARG --abl_head_name random \
        --num_ablate 0 --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
elif [ -n "$ABL_HEAD" ]; then
    python src/evaluate_icl_score.py --model_name "$MODEL" $CKPT_ARG --abl_head_name "$ABL_HEAD" \
        --num_ablate "$ABL" --exclude_other_heads "$EXCL" --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
else
    for HEAD in induction fv random; do
        python src/evaluate_icl_score.py --model_name "$MODEL" $CKPT_ARG --abl_head_name "$HEAD" \
            --num_ablate "$ABL" --exclude_other_heads "$EXCL" --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
    done
fi
