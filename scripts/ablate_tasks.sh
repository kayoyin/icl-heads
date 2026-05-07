#!/bin/bash
# Few-shot ICL accuracy under attention-head ablation (Section 4).
# Required env vars:
#   MODEL          - key in MODEL_NAME_DICT
#   ABL            - fraction of heads to ablate (e.g. 0.01, 0.05, 0.2). Use 0 for clean.
# Optional env vars:
#   CKPT           - Pythia checkpoint step
#   ABL_HEAD       - induction|fv|random (default: cycles through induction, fv, and random)
#   EXCL           - 1 to use ablation-with-exclusion (default: 0)
#   ZERO           - 1 to zero-ablate instead of mean-ablate (default: 0)
#   RAND_ACT       - 1 to shuffle the activation tensor before ablation (default: 0)
#   SEED, FORCE, OUT - same as other scripts
set -e

MODEL=${MODEL:-160m}
ABL=${ABL:-0}
EXCL=${EXCL:-0}
ZERO=${ZERO:-0}
RAND_ACT=${RAND_ACT:-0}
SEED=${SEED:-42}
FORCE=${FORCE:-0}
OUT=${OUT:-./outputs}

CKPT_ARG=""
if [ -n "$CKPT" ]; then
    CKPT_ARG="--ckpt $CKPT"
fi

if [ "$ABL" = "0" ]; then
    python src/ablate.py --model_name "$MODEL" $CKPT_ARG --abl_head_name random \
        --num_ablate 0 --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
elif [ -n "$ABL_HEAD" ]; then
    python src/ablate.py --model_name "$MODEL" $CKPT_ARG --abl_head_name "$ABL_HEAD" \
        --num_ablate "$ABL" --random_ablate "$RAND_ACT" --zero_ablate "$ZERO" \
        --exclude_other_heads "$EXCL" --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
else
    for HEAD in induction fv random; do
        python src/ablate.py --model_name "$MODEL" $CKPT_ARG --abl_head_name "$HEAD" \
            --num_ablate "$ABL" --random_ablate "$RAND_ACT" --zero_ablate "$ZERO" \
            --exclude_other_heads "$EXCL" --seed "$SEED" --force "$FORCE" --save_path_root "$OUT"
    done
fi
