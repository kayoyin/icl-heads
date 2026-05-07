"""
Compute induction scores for every attention head in a model.

Induction score is measured on synthetic sequences of repeated random tokens,
following the definition in Olsson et al. (2022). For each head we report the
mean attention paid to the token following a previous occurrence, averaged over
random sequences.
"""
import argparse
import os
import pickle

import torch
from transformer_lens import HookedTransformer
from transformer_lens.head_detector import detect_head

from utils.model_utils import MODEL_NAME_DICT, load_gpt_model_and_tokenizer, set_seed


def generate_repeated_random_tokens(model, batch=1000, seq_len=50, seed=0):
    """Generate `batch` sequences of the form [BOS, r_1..r_n, r_1..r_n]."""
    set_seed(seed)
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    rep_tokens = torch.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(model.cfg.device)
    return rep_tokens


def find_induction_heads(model, batch=1000, seq_len=50, seed=0):
    """Compute the induction score of every head on repeated random sequences."""
    rep_tokens = generate_repeated_random_tokens(model, batch, seq_len, seed)
    prompts = [model.tokenizer.decode(x) for x in rep_tokens]

    head_scores = detect_head(
        model,
        prompts,
        "induction_head",
        exclude_bos=False,
        exclude_current_token=False,
        error_measure="abs",
    )

    results = {}
    for layer, layer_scores in enumerate(head_scores):
        for head, score in enumerate(layer_scores):
            results[f"{layer}.{head}"] = score.item()
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute induction scores for every attention head.")
    parser.add_argument("--model_name", required=True, help="Model key in MODEL_NAME_DICT (e.g. '160m', 'gpt2', '7b').")
    parser.add_argument("--ckpt", type=int, default=None, help="Optional Pythia training checkpoint step.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path_root", default="./outputs", help="Directory under which to write heads/<model>/induction.pkl.")
    parser.add_argument("--force", type=int, default=0, help="Overwrite existing output if 1.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAME_DICT[args.model_name]
    set_seed(args.seed)

    short_name = model_name.split("/")[-1].replace("-deduped", "")
    if args.ckpt is not None:
        short_name = f"{short_name}-{args.ckpt}"
    output_path = os.path.join(args.save_path_root, "heads", short_name)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "induction.pkl")

    if os.path.exists(output_file) and not bool(args.force):
        print(f"{output_file} already exists; pass --force 1 to overwrite.")
        return

    if "llama" in model_name.lower():
        hfmodel, tokenizer, _ = load_gpt_model_and_tokenizer(model_name, device=device)
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hfmodel,
            torch_dtype=torch.bfloat16,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
        )
    elif args.ckpt is not None:
        model, _, _ = load_gpt_model_and_tokenizer(model_name, device=device, ckpt=args.ckpt)
    else:
        model, _, _ = load_gpt_model_and_tokenizer(model_name, device=device)
    model.eval()

    results = find_induction_heads(model, seed=args.seed)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
