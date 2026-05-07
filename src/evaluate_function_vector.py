"""
Evaluate the prevalence of the FV mechanism in a given model (Section A.6).

For each ICL task we:
  1. extract a function vector from the top-`num_fv_heads` heads ranked by FV score;
  2. compare model accuracy on the clean prompt, on a shuffled-label prompt,
     and on a shuffled-label prompt with the FV (or a random-head FV) added at
     layer |L|/3.
"""
import argparse
import json
import os

import numpy as np
import torch

from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention, compute_dataset_baseline
from utils.extract_utils import compute_function_vector
from utils.model_utils import MODEL_NAME_DICT, load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset

# Same set of held-out tasks used elsewhere in the paper.
FV_PAPER_EVAL_TASKS = [
    "ag_news_val", "antonym_val", "capitalize_val", "capitalize_first_letter_val",
    "capitalize_second_letter_val", "commonsense_qa_val",
    "english-french_val", "english-german_val", "english-spanish_val",
    "landmark-country_val", "lowercase_first_letter_val", "national_parks_val",
    "park-country_val", "person-instrument_val", "person-occupation_val",
    "product-company_val", "sentiment_val", "synonym_val",
    "adjective_v_verb_3_val", "adjective_v_verb_5_val",
    "animal_v_object_3_val", "animal_v_object_5_val",
    "choose_first_of_3_val", "choose_first_of_5_val",
    "choose_last_of_3_val", "choose_last_of_5_val",
    "choose_middle_of_3_val", "choose_middle_of_5_val",
    "color_v_animal_3_val", "color_v_animal_5_val",
    "concept_v_object_3_val", "concept_v_object_5_val",
    "conll2003_location_val", "conll2003_organization_val", "conll2003_person_val",
    "fruit_v_animal_3_val", "fruit_v_animal_5_val",
    "object_v_concept_3_val", "object_v_concept_5_val",
    "verb_v_adjective_3_val", "verb_v_adjective_5",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate FV-based task execution (Section A.6).")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--model_name", required=True, help="Model key in MODEL_NAME_DICT.")
    parser.add_argument("--save_path_root", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--metric", default="f1_score")
    parser.add_argument("--force", type=int, default=0)
    parser.add_argument("--randomize", type=int, default=0,
                        help="If 1, sample heads at random instead of by FV score (sanity baseline).")
    parser.add_argument("--num_fv_heads", type=float, default=0.01,
                        help="Fraction of all heads to use when constructing the FV.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAME_DICT[args.model_name]
    force = bool(int(args.force))
    randomize = bool(int(args.randomize))

    short_name = model_name.split("/")[-1].replace("-deduped", "")
    head_path = f"{args.save_path_root}/heads/{short_name}"
    score_path = f"{args.save_path_root}/fv_scores/{short_name}"
    os.makedirs(score_path, exist_ok=True)

    torch.set_grad_enabled(False)
    print(f"Loading {model_name}")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)
    set_seed(args.seed)

    datasets = [args.dataset_name] if args.dataset_name else FV_PAPER_EVAL_TASKS

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name, seed=args.seed)
        baseline_file = f"{score_path}/{dataset_name}_clean.json"
        zs_file = f"{score_path}/{dataset_name}_zs{'_random' if randomize else ''}.json"
        fs_shuffled_file = f"{score_path}/{dataset_name}_shuffled{'_random' if randomize else ''}.json"

        if (
            os.path.exists(zs_file)
            and os.path.exists(fs_shuffled_file)
            and os.path.exists(baseline_file)
            and not force
        ):
            print(f"{dataset_name}: results exist; skipping.")
            continue

        # Filter test set to examples the model gets correct via plain ICL.
        fs_results_file = f"{score_path}/{dataset_name}_fs_results.json"
        if os.path.exists(fs_results_file):
            with open(fs_results_file) as f:
                fs_results = json.load(f)
            filter_set = np.where(np.array(fs_results["clean_rank_list"]) == 0)[0]
        else:
            set_seed(args.seed)
            fs_results = n_shot_eval_no_intervention(
                dataset=dataset, n_shots=args.n_shots, model=model, model_config=model_config,
                tokenizer=tokenizer, compute_ppl=True,
            )
            filter_set = np.where(np.array(fs_results["clean_rank_list"]) == 0)[0]

        with open(fs_results_file, "w") as f:
            json.dump(fs_results, f, indent=2)

        set_seed(args.seed)
        mean_activations = torch.load(f"{head_path}/mean_head_activations.pt")
        indirect_effect = torch.load(f"{head_path}/indirect_effect.pt")
        if randomize:
            indirect_effect = torch.rand_like(indirect_effect)

        n_top_heads = max(1, indirect_effect.shape[1] * indirect_effect.shape[2] * args.num_fv_heads)
        n_top_heads = int(np.ceil(n_top_heads))

        if not os.path.exists(baseline_file) or force:
            baseline_results = compute_dataset_baseline(
                dataset, model, model_config, tokenizer, n_shots=args.n_shots, seed=args.seed
            )
            with open(baseline_file, "w") as f:
                json.dump(baseline_results, f, indent=2)

        fv, _ = compute_function_vector(
            mean_activations, indirect_effect, model, model_config=model_config, n_top_heads=n_top_heads
        )
        eval_edit_layer = indirect_effect.shape[1] // 3

        if not os.path.exists(zs_file) or force:
            set_seed(args.seed)
            zs_results = n_shot_eval(
                dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=0,
                model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set,
            )
            with open(zs_file, "w") as f:
                json.dump(zs_results, f, indent=2)

        if not os.path.exists(fs_shuffled_file) or force:
            set_seed(args.seed)
            fs_shuffled_results = n_shot_eval(
                dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=args.n_shots,
                model=model, model_config=model_config, tokenizer=tokenizer,
                filter_set=filter_set, shuffle_labels=True,
            )
            with open(fs_shuffled_file, "w") as f:
                json.dump(fs_shuffled_results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
