# Which Attention Heads Matter for In-Context Learning?

Code and data for **Yin & Steinhardt (2025), *"Which Attention Heads Matter for In-Context Learning?"***
([arXiv:2502.14010](https://arxiv.org/abs/2502.14010)).

We compare two mechanisms proposed for in-context learning (ICL) in transformer
language models — **induction heads** (Olsson et al., 2022) and **function
vector (FV) heads** (Todd et al., 2024) — across 12 decoder-only models
ranging from 70M to 7B parameters. Through systematic ablation studies and
training-dynamics analysis we find that:

* FV heads are the primary drivers of few-shot ICL accuracy, especially in
  larger models.
* Induction and FV heads are largely distinct but their scores are correlated.
* During Pythia training, many FV heads first emerge as induction heads before
  transitioning to the FV mechanism.

## Repository layout

```
src/
  find_induction_heads.py     # Induction score on synthetic repeated sequences
  find_fv_heads.py            # FV score via causal mediation (Todd et al. 2024)
  ablate.py                   # Few-shot ICL accuracy under head ablation (Sec. 4)
  evaluate_icl_score.py       # Token-loss difference under head ablation (Sec. 4)
  evaluate_function_vector.py # FV-based task execution (Sec. A.6)
  utils/
    model_utils.py            # MODEL_NAME_DICT and HF model loading
    prompt_utils.py           # ICL prompt construction and dataset loading
    extract_utils.py          # Mean head activations, FV computation
    intervention_utils.py     # TraceDict-based activation interventions
    ablate_utils.py           # Head-selection logic and TL-based ablation hooks
    eval_utils.py             # Evaluation helpers (F1, n-shot eval, etc.)
scripts/
  find_induction_heads.sh     # One-shot launcher for find_induction_heads.py
  find_fv_heads.sh            # …and find_fv_heads.py
  ablate_tasks.sh             # …and ablate.py
  eval_icl.sh                 # …and evaluate_icl_score.py
  eval_fv.sh                  # …and evaluate_function_vector.py
dataset_files/
  abstractive/                # Tasks from Todd et al. (2024)
  extractive/                 # Tasks from Todd et al. (2024)
  new_tasks/                  # Tasks introduced in this paper
  generate/                   # Scripts used to construct the datasets
notebooks/
  plots.ipynb                 # Figures from the paper
run.sh                        # Reference experiment launcher
requirements.txt
```

## Setup

```bash
git clone https://github.com/kayoyin/icl-heads.git
cd icl-heads
pip install -r requirements.txt
```

By default datasets are loaded from `<repo>/dataset_files`. To point elsewhere
set `ICL_HEADS_DATA_DIR=/path/to/dataset_files` before launching the scripts.

Llama-2 weights require accepting the licence on Hugging Face; log in with
`huggingface-cli login` first if you plan to run those models.

## Reproducing the experiments

The driver scripts under `scripts/` accept env vars (`MODEL`, `CKPT`, `ABL`,
`EXCL`, `OUT`, …). The valid `MODEL` keys are listed in
`src/utils/model_utils.py:MODEL_NAME_DICT` (e.g. `70m`, `160m`, `1b`, `7b`,
`gpt2-large`).

### 1. Compute induction and FV scores

```bash
MODEL=160m bash scripts/find_induction_heads.sh   # writes outputs/heads/pythia-160m/induction.pkl
MODEL=160m bash scripts/find_fv_heads.sh          # writes fv.pkl, mean_head_activations.pt, indirect_effect.pt
```

For training-dynamics analysis, pass `CKPT=<step>`:

```bash
MODEL=160m CKPT=1000 bash scripts/find_induction_heads.sh
MODEL=160m CKPT=1000 bash scripts/find_fv_heads.sh
```

### 2. Few-shot ICL accuracy under ablation (Section 4)

```bash
# Top row of Fig. 4: ablate top fraction of induction / FV / random heads.
MODEL=6.9b ABL=0.05 EXCL=0 bash scripts/ablate_tasks.sh

# Middle row: same, but exclude top-2% of the *other* mechanism's heads.
MODEL=6.9b ABL=0.05 EXCL=1 bash scripts/ablate_tasks.sh
```

`ABL_HEAD=induction|fv|random` selects a single ranking; otherwise the
script iterates over all three. Outputs are written under
`outputs/scores/<model>/...` and `outputs/preds/<model>/...`.

### 3. Token-loss difference under ablation (Section 4, bottom row of Fig. 4)

```bash
MODEL=6.9b ABL=0.05 EXCL=1 bash scripts/eval_icl.sh
```

Sequences are sampled from the Pile via TransformerLens; the script writes
`<head>_<frac>_50.txt`, `<head>_<frac>_500.txt`, and `<head>_<frac>_score.txt`.

### 4. FV-based task execution (Section A.6, Fig. 14)

```bash
MODEL=6.9b M=0.02 bash scripts/eval_fv.sh
```

Runs both the actual FV (`--randomize 0`) and a random-head baseline
(`--randomize 1`) for each task in `FV_PAPER_EVAL_TASKS`.

### 5. Reproduce the paper figures

`notebooks/plots.ipynb` consumes the per-experiment outputs above and emits the
figures shown in the paper.

`run.sh` lists the exact loops used to generate every figure.

## Citation

```bibtex
@inproceedings{yin2025which,
  title     = {Which Attention Heads Matter for In-Context Learning?},
  author    = {Yin, Kayo and Steinhardt, Jacob},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://openreview.net/forum?id=C7XmEByCFv}
}
```

## Acknowledgements

The FV-extraction code in `src/utils/{extract_utils,intervention_utils,eval_utils,prompt_utils}.py`
and the ICL task data in `dataset_files/abstractive` and `dataset_files/extractive`
are adapted from Todd et al. (2024)'s
[function_vectors](https://github.com/ericwtodd/function_vectors) repo.
The binding tasks under `dataset_files/new_tasks/bind_*` are from Feng &
Steinhardt (2024)'s [LM_binding](https://github.com/jiahai-feng/binding-iclr).
Induction-score computation uses
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
