"""
Few-shot ICL accuracy evaluation under attention-head ablation.

For each task in the held-out evaluation set we generate predictions on a
fixed number of ICL prompts, optionally replacing the activations of selected
attention heads with the task-mean activation (or with a zero / shuffled
substitute). The resulting per-task F1 scores match the curves in section 4
of the paper.
"""
import argparse
import os
import re

import numpy as np
import torch
from baukit import TraceDict

from utils.ablate_utils import get_heads_to_ablate
from utils.eval_utils import f1_score, parse_generation
from utils.intervention_utils import replace_activation_w_avg
from utils.model_utils import MODEL_NAME_DICT, load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import (
    get_dummy_token_labels,
    get_token_meta_labels,
    load_dataset,
    word_pairs_to_prompt_data,
)

# Held-out evaluation tasks (Section 4). These are *not* used to compute FV
# scores (which are estimated on DEFAULT_FV_TRAIN_TASKS in find_fv_heads.py),
# so there is no leakage between scoring and evaluation.
FV_PAPER_TEST = [
    "ag_news_val", "capitalize_first_letter_val", "conll2003_organization_val",
    "national_parks_val", "park-country_val", "person-occupation_val",
    "animal_v_object_3_val", "choose_first_of_5_val", "english-german_val",
    "person-instrument_val", "commonsense_qa_val",
]

# New evaluation tasks introduced in this paper (bold rows in Table 3).
NEW_EVAL_TASKS = [
    "capital_index_val", "capitalize_second_letter_train", "abstract_clf_val",
    "french-english_val", "bind_fruit_val", "bind_capital_par_val",
    "bind_capital_val", "bind_shape_val",
]


def generate_predictions(
    model, model_config, tokenizer, dataset, avg_activations, n_shots, n_trials,
    heads_to_ablate, random_ablate=False,
):
    """Generate predictions on `n_trials` ICL prompts and score them with F1."""
    is_llama = "llama" in model_config["name_or_path"].lower()
    prepend_bos = not is_llama
    refs, preds, scores = [], [], []
    device = model.device

    if random_ablate:
        # Shuffle the activation tensor so that interventions inject random-but-in-distribution
        # values rather than task-mean values. Used as a sanity baseline.
        shape = avg_activations.shape
        flattened = avg_activations.view(-1)
        random_indices = torch.randperm(flattened.size(0))
        avg_activations = flattened[random_indices].view(shape)

    for i in range(n_trials):
        word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        word_pairs_test = dataset["valid"][i]
        prompt_data = word_pairs_to_prompt_data(
            word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos
        )

        query_target_pair = prompt_data["query_target"]
        query = query_target_pair["input"]
        token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
        sentences = [prompt_string]

        get_dummy_token_labels(n_shots, tokenizer=tokenizer)  # ensures tokenizer is warmed up

        class_regex = "query_predictive_token"
        reg_class_match = re.compile(f"^{class_regex}$")
        class_token_inds = [x[0] for x in token_labels if reg_class_match.match(x[2])]

        inputs = tokenizer(sentences, return_tensors="pt").to(device)
        ref = prompt_data["query_target"]["output"].strip()
        refs.append(ref)

        MAX_NEW_TOKENS = 5

        if len(heads_to_ablate) == 0:
            output = model.generate(
                inputs.input_ids, top_p=0.9, temperature=0.1, max_new_tokens=MAX_NEW_TOKENS
            )
            output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        else:
            head_hook_layer = [model_config["attn_hook_names"][layer] for layer, _ in heads_to_ablate]
            intervention_locations = [
                (layer, head_n, t) for t in class_token_inds for layer, head_n in heads_to_ablate
            ]
            intervention_fn = replace_activation_w_avg(
                layer_head_token_pairs=intervention_locations,
                avg_activations=avg_activations,
                model=model, model_config=model_config,
                batched_input=False, idx_map=None, last_token_only=True,
            )
            with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn):
                output = model.generate(
                    inputs.input_ids, top_p=0.9, temperature=0.1, max_new_tokens=MAX_NEW_TOKENS
                )
                output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])

        pred, score = parse_generation(output, [ref], f1_score)
        try:
            preds.append(str(pred))
        except Exception:
            preds.append(pred[0])
        scores.append(str(score))
    return refs, preds, scores


def _build_paths(save_path_root, model_short_name, ckpt, random_ablate, zero_ablate, exclude_other_heads, excl_num, n_shots):
    suffix = f"-{ckpt}" if ckpt is not None else ""
    base_model = f"{model_short_name}{suffix}"
    head_path = f"{save_path_root}/heads/{base_model}"
    refs_path = f"{save_path_root}/refs/{base_model}"

    if random_ablate:
        score_suffix = "_random"
    elif zero_ablate:
        score_suffix = "_zero"
    else:
        score_suffix = ""

    score_path = f"{save_path_root}/scores/{base_model}{score_suffix}"
    preds_path = f"{save_path_root}/preds/{base_model}{score_suffix}"

    if exclude_other_heads:
        n = int(excl_num * 100)
        score_path += f"_excltop{n}"
        preds_path += f"_excltop{n}"

    if n_shots == 0:
        score_path += "_zero_shot"
        preds_path += "_zero_shot"

    return head_path, refs_path, score_path, preds_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate few-shot ICL accuracy under head ablation.")
    parser.add_argument("--model_name", required=True, help="Model key in MODEL_NAME_DICT.")
    parser.add_argument("--ckpt", type=int, default=None, help="Optional Pythia training checkpoint step.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path_root", default="./outputs",
                        help="Same path used by find_*_heads.py; expects heads/<model>/ to exist.")
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--abl_head_name", default="induction",
                        help="Which heads to rank when picking the top --num_ablate. Supported: induction|fv|random.")
    parser.add_argument("--num_ablate", type=float, default=0,
                        help="Number of heads to ablate. If <1, interpreted as a fraction of total heads.")
    parser.add_argument("--random_ablate", type=int, default=0,
                        help="If 1, shuffle the mean-activation tensor before ablation (sanity baseline).")
    parser.add_argument("--zero_ablate", type=int, default=0,
                        help="If 1, replace activations with zero instead of mean.")
    parser.add_argument("--exclude_other_heads", type=int, default=0,
                        help="If 1, exclude top --excl_num fraction of the *other* mechanism's heads (Sec. 4).")
    parser.add_argument("--excl_num", type=float, default=0.02)
    parser.add_argument("--force", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Override evaluation tasks; defaults to FV_PAPER_TEST + NEW_EVAL_TASKS.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAME_DICT[args.model_name]
    random_ablate = bool(int(args.random_ablate))
    zero_ablate = bool(int(args.zero_ablate))
    exclude_other_heads = bool(int(args.exclude_other_heads))
    force = bool(int(args.force))
    set_seed(args.seed)

    print(
        f"Ablating {args.abl_head_name} num={args.num_ablate} "
        f"random={random_ablate} zero={zero_ablate} exclude_other={exclude_other_heads}"
    )

    short_name = model_name.split("/")[-1].replace("-deduped", "")
    head_path, refs_path, score_path, preds_path = _build_paths(
        args.save_path_root, short_name, args.ckpt,
        random_ablate, zero_ablate, exclude_other_heads, args.excl_num, args.n_shots,
    )

    if args.dataset_name is None:
        datasets = FV_PAPER_TEST + NEW_EVAL_TASKS
    else:
        datasets = [args.dataset_name]

    torch.set_grad_enabled(False)
    print(f"Loading {model_name}")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, ckpt=args.ckpt)

    os.makedirs(score_path, exist_ok=True)
    os.makedirs(preds_path, exist_ok=True)
    os.makedirs(refs_path, exist_ok=True)

    heads_to_ablate = get_heads_to_ablate(
        head_path, args.abl_head_name, args.num_ablate, exclude_other_heads, args.excl_num
    )
    avg_activations = torch.load(f"{head_path}/mean_head_activations.pt")
    if zero_ablate:
        avg_activations = torch.zeros_like(avg_activations)

    for dataset_name in datasets:
        print(f"=== Task: {dataset_name} ===")
        save_path = f'{dataset_name.replace("_val", "")}_{args.abl_head_name}_{args.num_ablate}'
        if os.path.exists(f"{score_path}/{save_path}.txt") and not force:
            print(f"Skipping {dataset_name}: {save_path}.txt already exists.")
            continue

        dataset = load_dataset(dataset_name, test_size=0.9)
        refs, preds, scores = generate_predictions(
            model, model_config, tokenizer, dataset, avg_activations,
            args.n_shots, args.n_trials, heads_to_ablate, random_ablate=random_ablate,
        )

        with open(f'{refs_path}/{dataset_name.replace("_val", "")}.txt', "w", encoding="utf-8") as f:
            f.write("\n".join(refs))
        with open(f"{preds_path}/{save_path}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(preds))
        with open(f"{score_path}/{save_path}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(scores))


if __name__ == "__main__":
    main()
