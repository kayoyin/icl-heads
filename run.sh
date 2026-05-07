#!/bin/bash
# Reference launcher for the experiments in
#   "Which Attention Heads Matter for In-Context Learning?" (Yin & Steinhardt, 2025).
#
# Each block below describes one figure / table from the paper and shows how to
# regenerate the underlying numbers. The scripts under scripts/ accept env vars
# (MODEL, CKPT, ABL, EXCL, ...); see those files for the full list.
#
# These commands are *not* meant to all run in sequence on one machine — they
# are organised by experiment so you can launch the subset you need (e.g. by
# wrapping each call in `sbatch` for a SLURM cluster).
set -euo pipefail

OUT=${OUT:-./outputs}
SEED=${SEED:-42}
FORCE=${FORCE:-0}

ALL_MODELS=(70m 160m 410m 1b 1.4b 2.8b 6.9b gpt2 gpt2-medium gpt2-large gpt2-xl 7b)
PYTHIA_CKPTS=(1 64 256 1000 4000 16000 64000 143000)

# ---------------------------------------------------------------------------
# 1. Find induction heads and FV heads (Section 3, fully-trained models)
# ---------------------------------------------------------------------------
# for M in "${ALL_MODELS[@]}"; do
#   MODEL=$M FORCE=$FORCE SEED=$SEED OUT=$OUT bash scripts/find_induction_heads.sh
#   MODEL=$M FORCE=$FORCE SEED=$SEED OUT=$OUT bash scripts/find_fv_heads.sh
# done

# ---------------------------------------------------------------------------
# 2. Find induction/FV heads across Pythia training checkpoints (Section 5)
# ---------------------------------------------------------------------------
# for M in 70m 160m 410m 1b 1.4b 2.8b 6.9b; do
#   for CKPT in "${PYTHIA_CKPTS[@]}"; do
#     MODEL=$M CKPT=$CKPT FORCE=$FORCE SEED=$SEED OUT=$OUT bash scripts/find_induction_heads.sh
#     MODEL=$M CKPT=$CKPT FORCE=$FORCE SEED=$SEED OUT=$OUT bash scripts/find_fv_heads.sh
#   done
# done

# ---------------------------------------------------------------------------
# 3. Few-shot ICL accuracy under ablation (Section 4, top row of Figure 4)
# ---------------------------------------------------------------------------
# for M in "${ALL_MODELS[@]}"; do
#   for ABL in 0 0.01 0.03 0.05 0.07 0.09 0.15 0.2; do
#     MODEL=$M ABL=$ABL EXCL=0 FORCE=$FORCE SEED=$SEED OUT=$OUT bash scripts/ablate_tasks.sh
#   done
# done

# ---------------------------------------------------------------------------
# 4. Few-shot ICL accuracy with exclusion (Section 4, middle row of Figure 4)
# ---------------------------------------------------------------------------
# for M in "${ALL_MODELS[@]}"; do
#   for ABL in 0.01 0.03 0.05 0.07 0.09 0.15 0.2; do
#     MODEL=$M ABL=$ABL EXCL=1 FORCE=$FORCE SEED=$SEED OUT=$OUT bash scripts/ablate_tasks.sh
#   done
# done

# ---------------------------------------------------------------------------
# 5. Token-loss difference under ablation (Section 4, bottom row of Figure 4)
# ---------------------------------------------------------------------------
# for M in "${ALL_MODELS[@]}"; do
#   for ABL in 0 0.01 0.03 0.05 0.07 0.09 0.15 0.2; do
#     MODEL=$M ABL=$ABL EXCL=1 FORCE=$FORCE SEED=$SEED OUT=$OUT bash scripts/eval_icl.sh
#   done
# done

# ---------------------------------------------------------------------------
# 6. Evaluate FV-based task execution (Section A.6 / Figure 14)
# ---------------------------------------------------------------------------
# for M in "${ALL_MODELS[@]}"; do
#   MODEL=$M M=0.02 FORCE=$FORCE SEED=$SEED OUT=$OUT bash scripts/eval_fv.sh
# done
