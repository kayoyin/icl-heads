"""
Compute function vector (FV) scores for every attention head in a model via
the causal mediation procedure of Todd et al. (2024).

For each ICL task, we:
  1. compute the mean activation of every attention head over clean ICL prompts;
  2. construct corrupted prompts (shuffled labels) with the same inputs;
  3. measure each head's contribution to recovering the correct output when its
     activation on the corrupted prompt is overwritten with its task-mean.

Final scores are averaged across tasks. We also save the underlying mean-head
activations and the per-trial indirect-effect tensor, both of which are reused
by `evaluate_function_vector.py` and `ablate.py`.
"""
import argparse
import json
import os
import pickle
import re

import numpy as np
import torch
from baukit import TraceDict
from tqdm import tqdm

from utils.extract_utils import get_mean_head_activations
from utils.intervention_utils import replace_activation_w_avg
from utils.model_utils import MODEL_NAME_DICT, load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import (
    compute_duplicated_labels,
    get_dummy_token_labels,
    get_token_meta_labels,
    load_dataset,
    update_idx_map,
    word_pairs_to_prompt_data,
)

# Tasks used to estimate the FV score; the held-out evaluation tasks are kept
# separate in `ablate.py` so that there is no leakage between scoring and eval.
DEFAULT_FV_TRAIN_TASKS = [
    "product-company_train", "landmark-country_train", "sentiment_train", "antonym_train",
    "concept_v_object_5_train", "concept_v_object_3_train", "fruit_v_animal_5_train",
    "synonym_train", "lowercase_first_letter_train", "object_v_concept_5_train",
    "choose_middle_of_5_train", "conll2003_person_train", "color_v_animal_5_train",
    "choose_last_of_3_train", "adjective_v_verb_3_train", "verb_v_adjective_5",
    "animal_v_object_5_val", "conll2003_location_train", "adjective_v_verb_5_train",
    "color_v_animal_3_train", "choose_middle_of_3_train", "object_v_concept_3_train",
    "capitalize_train", "fruit_v_animal_3_train", "choose_last_of_5_train",
    "choose_first_of_3_train", "verb_v_adjective_3_train", "english-spanish_train",
    "english-french_train",
]


def activation_replacement_per_class_intervention(
    prompt_data, avg_activations, dummy_labels, model, model_config, tokenizer, last_token_only=True
):
    """Sweep over (layer, head) and measure the indirect effect on the correct output."""
    device = model.device
    query_target_pair = prompt_data["query_target"]
    query = query_target_pair["input"]
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query=query)

    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
    idx_map = update_idx_map(idx_map, idx_avg)

    sentences = [prompt_string]
    tokens_of_interest = [query_target_pair["output"]]
    if "llama" in model_config["name_or_path"]:
        ts = tokenizer(tokens_of_interest, return_tensors="pt").input_ids.squeeze()
        # Avoid SentencePiece spacing issues.
        if tokenizer.decode(ts[1]) == "" or ts[1] == 29871:
            token_id_of_interest = ts[2]
        else:
            token_id_of_interest = ts[1]
    else:
        token_id_of_interest = tokenizer(tokens_of_interest).input_ids[0][:1]

    inputs = tokenizer(sentences, return_tensors="pt").to(device)

    if last_token_only:
        token_classes = ["query_predictive"]
        token_classes_regex = ["query_predictive_token"]
    else:
        token_classes = [
            "demonstration", "label", "separator", "predictive", "structural", "end_of_example",
            "query_demonstration", "query_structural", "query_separator", "query_predictive",
        ]
        token_classes_regex = [
            r"demonstration_[\d]{1,}_token", r"demonstration_[\d]{1,}_label_token", "separator_token",
            "predictive_token", "structural_token", "end_of_example_token",
            "query_demonstration_token", "query_structural_token", "query_separator_token",
            "query_predictive_token",
        ]

    indirect_effect_storage = torch.zeros(model_config["n_layers"], model_config["n_heads"], len(token_classes))

    clean_output = model(**inputs).logits[:, -1, :]
    clean_probs = torch.softmax(clean_output[0], dim=-1)

    for layer in range(model_config["n_layers"]):
        head_hook_layer = [model_config["attn_hook_names"][layer]]
        for head_n in range(model_config["n_heads"]):
            for i, (token_class, class_regex) in enumerate(zip(token_classes, token_classes_regex)):
                reg_class_match = re.compile(f"^{class_regex}$")
                class_token_inds = [x[0] for x in token_labels if reg_class_match.match(x[2])]

                intervention_locations = [(layer, head_n, t) for t in class_token_inds]
                intervention_fn = replace_activation_w_avg(
                    layer_head_token_pairs=intervention_locations,
                    avg_activations=avg_activations,
                    model=model, model_config=model_config,
                    batched_input=False, idx_map=idx_map, last_token_only=last_token_only,
                )
                with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn):
                    output = model(**inputs).logits[:, -1, :]

                intervention_probs = torch.softmax(output, dim=-1)
                indirect_effect_storage[layer, head_n, i] = (
                    (intervention_probs - clean_probs)
                    .index_select(1, torch.LongTensor(token_id_of_interest).to(device).squeeze())
                    .squeeze()
                )

    return indirect_effect_storage


def compute_indirect_effect(
    dataset, mean_activations, model, model_config, tokenizer,
    n_shots=10, n_trials=25, last_token_only=True, prefixes=None, separators=None,
    filter_set=None, save_path_root=None,
):
    """Average the indirect effect of every head across `n_trials` shuffled-label prompts."""
    n_test_examples = 1
    if prefixes is not None and separators is not None:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes, separators=separators)
    else:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer)

    is_llama = "llama" in model_config["name_or_path"].lower()
    prepend_bos = not is_llama

    if last_token_only:
        indirect_effect = torch.zeros(n_trials, model_config["n_layers"], model_config["n_heads"])
    else:
        indirect_effect = torch.zeros(n_trials, model_config["n_layers"], model_config["n_heads"], 10)

    if filter_set is None:
        filter_set = np.arange(len(dataset["valid"]))

    for i in tqdm(range(n_trials), total=n_trials):
        word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        word_pairs_test = dataset["valid"][np.random.choice(filter_set, n_test_examples, replace=False)]
        kwargs = dict(query_target_pair=word_pairs_test, shuffle_labels=True, prepend_bos_token=prepend_bos)
        if prefixes is not None and separators is not None:
            kwargs.update(prefixes=prefixes, separators=separators)
        prompt_data_random = word_pairs_to_prompt_data(word_pairs, **kwargs)

        ind_effects = activation_replacement_per_class_intervention(
            prompt_data=prompt_data_random,
            avg_activations=mean_activations,
            dummy_labels=dummy_gt_labels,
            model=model, model_config=model_config, tokenizer=tokenizer,
            last_token_only=last_token_only,
        )
        indirect_effect[i] = ind_effects.squeeze()

        if i % 10 == 0 and save_path_root is not None:
            torch.save(indirect_effect, f"{save_path_root}/indirect_effect.pt")

    return indirect_effect


def ie_to_dict(indirect_effect):
    """Convert (trials, layers, heads) tensor into a {'L.H': score} dict averaged over trials."""
    indirect_effect_dict = {}
    indirect_effect = torch.mean(indirect_effect, dim=0)
    for i in range(indirect_effect.shape[0]):
        for j in range(indirect_effect.shape[1]):
            indirect_effect_dict[f"{i}.{j}"] = indirect_effect[i, j].item()
    return indirect_effect_dict


def main():
    parser = argparse.ArgumentParser(description="Compute FV scores for every attention head.")
    parser.add_argument("--model_name", required=True, help="Model key in MODEL_NAME_DICT.")
    parser.add_argument("--ckpt", type=int, default=None, help="Optional Pythia training checkpoint step.")
    parser.add_argument("--save_path_root", default="./outputs",
                        help="Directory under which to write heads/<model>/{fv.pkl,mean_head_activations.pt,indirect_effect.pt}.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--last_token_only", type=bool, default=True)
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument("--force", type=int, default=0)
    args = parser.parse_args()

    model_name = MODEL_NAME_DICT[args.model_name]
    set_seed(args.seed)
    force = bool(int(args.force))

    short_name = model_name.split("/")[-1].replace("-deduped", "")
    if args.ckpt is not None:
        short_name = f"{short_name}-{args.ckpt}"
    output_path = os.path.join(args.save_path_root, "heads", short_name)
    os.makedirs(output_path, exist_ok=True)

    torch.set_grad_enabled(False)
    print(f"Loading {model_name}")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=args.device, ckpt=args.ckpt)

    mean_activations_file = os.path.join(output_path, "mean_head_activations.pt")
    indirect_effect_file = os.path.join(output_path, "indirect_effect.pt")
    fv_file = os.path.join(output_path, "fv.pkl")

    mean_indirect_effects = None
    count = 0

    for dataset_name in DEFAULT_FV_TRAIN_TASKS:
        print(f"=== Task: {dataset_name} ===")
        dataset = load_dataset(dataset_name, seed=0)

        per_task_mean_file = os.path.join(output_path, f"{dataset_name}_mean_head_activations.pt")
        per_task_ie_file = os.path.join(output_path, f"{dataset_name}_indirect_effect.pt")

        if not force and os.path.exists(per_task_mean_file):
            mean_activations = torch.load(per_task_mean_file)
        else:
            mean_activations = get_mean_head_activations(
                dataset, model=model, model_config=model_config, tokenizer=tokenizer,
                n_icl_examples=args.n_shots, N_TRIALS=args.n_trials,
                prefixes=args.prefixes, separators=args.separators,
            )
            torch.save(mean_activations, per_task_mean_file)

        if not force and os.path.exists(per_task_ie_file):
            indirect_effect = torch.load(per_task_ie_file)
        else:
            indirect_effect = compute_indirect_effect(
                dataset, mean_activations, model=model, model_config=model_config, tokenizer=tokenizer,
                n_shots=args.n_shots, n_trials=args.n_trials, last_token_only=args.last_token_only,
                prefixes=args.prefixes, separators=args.separators,
            )
            torch.save(indirect_effect, per_task_ie_file)

        if mean_indirect_effects is None:
            mean_indirect_effects = indirect_effect.clone()
            mean_activations_sum = mean_activations.clone()
        else:
            mean_indirect_effects += indirect_effect
            mean_activations_sum += mean_activations
        count += 1

    mean_indirect_effects /= count
    mean_activations_sum /= count

    torch.save(mean_activations_sum, mean_activations_file)
    torch.save(mean_indirect_effects, indirect_effect_file)
    with open(fv_file, "wb") as f:
        pickle.dump(ie_to_dict(mean_indirect_effects), f)
    print(f"Wrote FV scores to {fv_file}")


if __name__ == "__main__":
    main()
