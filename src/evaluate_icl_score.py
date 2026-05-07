"""
Token-loss difference (ICL score from Olsson et al., 2022) under head ablation.

We measure loss at the 50th vs. 500th token on Pile sequences and report the
difference, optionally with a subset of attention heads mean-ablated. Used in
section 4 to contrast token-loss difference against few-shot ICL accuracy.
"""
import argparse
import os

import torch
from transformer_lens import HookedTransformer, evals

from utils.ablate_utils import (
    get_heads_to_ablate,
    run_and_cache_model_random_tokens,
    run_loss_with_ablation,
)
from utils.model_utils import MODEL_NAME_DICT, load_gpt_model_and_tokenizer, set_seed


def in_context_learning_score(model, tokens):
    set_seed(42)
    loss_vec = model(tokens, return_type="loss", loss_per_token=True).detach().cpu()
    return loss_vec[..., 50].mean(), loss_vec[..., 500].mean(), (loss_vec[..., 50] - loss_vec[..., 500]).mean()


def in_context_learning_ablate_score(model, tokens, heads_to_ablate):
    set_seed(42)
    tokens = tokens[:, :505]
    seq_len = tokens.shape[1]
    _, _, random_cache = run_and_cache_model_random_tokens(model, seq_len, 1)
    random_cache.remove_batch_dim()
    loss_vec = run_loss_with_ablation(model, tokens, heads_to_ablate, random_cache).detach().cpu()
    model.reset_hooks()
    return loss_vec[..., 50].mean(), loss_vec[..., 500].mean(), (loss_vec[..., 50] - loss_vec[..., 500]).mean()


def _accumulate(model, pile_dataloader, batch_size, n_trials, scorer):
    """Average a per-batch scorer over `n_trials // batch_size` batches."""
    set_seed(42)
    loss_50 = loss_500 = score = 0.0
    num_batches = n_trials // batch_size
    for i, x in enumerate(pile_dataloader):
        tokens = x["tokens"].to(model.cfg.device)
        a, b, c = scorer(tokens)
        loss_50 += a.item()
        loss_500 += b.item()
        score += c.item()
        if i == num_batches:
            break
    return loss_50 / num_batches, loss_500 / num_batches, score / num_batches


def main():
    parser = argparse.ArgumentParser(description="Compute token-loss difference under head ablation.")
    parser.add_argument("--model_name", required=True, help="Model key in MODEL_NAME_DICT.")
    parser.add_argument("--ckpt", type=int, default=None, help="Optional Pythia training checkpoint step.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path_root", default="./outputs")
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--abl_head_name", default="induction", help="induction|fv|random")
    parser.add_argument("--num_ablate", type=float, default=0)
    parser.add_argument("--exclude_other_heads", type=int, default=0)
    parser.add_argument("--force", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_NAME_DICT[args.model_name]
    exclude_other_heads = bool(int(args.exclude_other_heads))
    force = bool(int(args.force))
    set_seed(args.seed)

    short_name = model_name.split("/")[-1].replace("-deduped", "")
    if args.ckpt is not None:
        short_name = f"{short_name}-{args.ckpt}"
    head_path = f"{args.save_path_root}/heads/{short_name}"
    score_path = f"{args.save_path_root}/icl-scores/{short_name}"
    if exclude_other_heads:
        score_path += "_excl"
    save_path = f"{args.abl_head_name}_{args.num_ablate}"

    if os.path.exists(f"{score_path}/{save_path}_score.txt") and not force:
        print("Token-loss difference already computed; pass --force 1 to overwrite.")
        return

    if "llama" in model_name:
        hfmodel, tokenizer, _ = load_gpt_model_and_tokenizer(model_name, device=device)
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hfmodel, torch_dtype=torch.bfloat16,
            fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer,
        )
    else:
        model = HookedTransformer.from_pretrained(model_name).to(device)

    pile_batch_size = 1
    pile_dataloader = evals.make_pile_data_loader(tokenizer=model.tokenizer, batch_size=pile_batch_size)

    os.makedirs(score_path, exist_ok=True)
    heads_to_ablate = get_heads_to_ablate(head_path, args.abl_head_name, args.num_ablate, exclude_other_heads)

    if args.num_ablate == 0:
        loss_50, loss_500, score = _accumulate(
            model, pile_dataloader, pile_batch_size, args.n_trials,
            lambda toks: in_context_learning_score(model, toks),
        )
    else:
        loss_50, loss_500, score = _accumulate(
            model, pile_dataloader, pile_batch_size, args.n_trials,
            lambda toks: in_context_learning_ablate_score(model, toks, heads_to_ablate),
        )

    with open(f"{score_path}/{save_path}_50.txt", "w") as f:
        f.write(str(loss_50))
    with open(f"{score_path}/{save_path}_500.txt", "w") as f:
        f.write(str(loss_500))
    with open(f"{score_path}/{save_path}_score.txt", "w") as f:
        f.write(str(score))
    print(f"Token-loss difference: {score:.4f} (loss@50={loss_50:.4f}, loss@500={loss_500:.4f})")


if __name__ == "__main__":
    main()
