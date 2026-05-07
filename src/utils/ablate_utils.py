"""
Utilities for selecting which attention heads to ablate, and for running
loss-only forward passes with TransformerLens hooks (used by
`evaluate_icl_score.py` to compute the token-loss difference under ablation).
"""
import pickle
import random
from functools import partial
from typing import Tuple

import torch
import transformer_lens.utils as tl_utils
from torchtyping import TensorType as TT
from transformer_lens import ActivationCache, HookedTransformer


def generate_random_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> torch.Tensor:
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    rep_tokens = torch.cat([prefix, rep_tokens_half], dim=-1).cuda()
    return rep_tokens


def run_and_cache_model_random_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, ActivationCache]:
    rep_tokens = generate_random_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache


def get_heads_to_ablate(head_path, abl_head_name, num_ablate, exclude_other_heads, excl_num=0.02):
    """
    Resolve `--abl_head_name` and `--num_ablate` into a concrete list of (layer, head)
    indices to ablate.

    Supported `abl_head_name` values:
      * "induction" - top heads by induction.pkl
      * "fv"        - top heads by fv.pkl
      * "random"    - randomly sampled heads (uses induction.pkl only to enumerate keys)

    If `exclude_other_heads` is set, the top `excl_num` fraction of heads under the
    *other* mechanism are removed from the ranking before selecting the top
    `num_ablate`. This implements the "ablation with exclusion" experiments from
    section 4 of the paper.
    """
    if num_ablate == 0:
        return []

    other_head_name = "fv" if "induction" in abl_head_name else "induction"

    if abl_head_name == "random":
        with open(f"{head_path}/induction.pkl", "rb") as f:
            tmp_heads = pickle.load(f)
        all_heads = {k: random.random() for k in tmp_heads.keys()}
    else:
        with open(f"{head_path}/{abl_head_name}.pkl", "rb") as f:
            all_heads = pickle.load(f)

    if exclude_other_heads:
        with open(f"{head_path}/{other_head_name}.pkl", "rb") as f:
            other_heads = pickle.load(f)

    all_heads = sorted(all_heads.keys(), key=lambda k: all_heads[k], reverse=True)

    if num_ablate < 1:
        num_ablate = int(num_ablate * len(all_heads))
    if num_ablate == 0:
        num_ablate = 1

    if exclude_other_heads:
        num_exclude = int(excl_num * len(all_heads))
        other_heads = sorted(other_heads.keys(), key=lambda k: other_heads[k], reverse=True)
        all_heads = [h for h in all_heads if h not in other_heads[:num_exclude]]

    heads_to_ablate = list(map(lambda x: list(map(int, x.split("."))), all_heads[: int(num_ablate)]))
    return heads_to_ablate


def head_ablation_hook(
    attn_result: TT["batch", "seq", "n_heads", "d_model"],
    hook,
    head_index_to_ablate,
    act_name,
    random_cache,
) -> TT["batch", "seq", "n_heads", "d_model"]:
    """Replace the activations of `head_index_to_ablate` with values from `random_cache`."""
    try:
        attn_result[:, :, head_index_to_ablate, :] = random_cache[act_name][
            :, head_index_to_ablate, :
        ].unsqueeze(0)
    except Exception:
        attn_result[:, :, head_index_to_ablate, :] = (
            random_cache[act_name][0, head_index_to_ablate, :].unsqueeze(0).unsqueeze(0)
        )
    return attn_result


def _add_fwd_bwd_hooks(model, fwd_hooks, bwd_hooks):
    for name, hook in fwd_hooks:
        if isinstance(name, str):
            model.mod_dict[name].add_hook(hook, dir="fwd")
    for name, hook in bwd_hooks:
        if isinstance(name, str):
            model.mod_dict[name].add_hook(hook, dir="bwd")


def generate_loss_with_hooks(
    model, tokens, fwd_hooks=(), bwd_hooks=(),
    reset_hooks_end=True, clear_contexts=False,
):
    """Forward pass returning per-token cross-entropy loss with the given hooks installed."""
    try:
        _add_fwd_bwd_hooks(model, fwd_hooks, bwd_hooks)
        return model(tokens, return_type="loss", loss_per_token=True)
    finally:
        if reset_hooks_end:
            model.reset_hooks(clear_contexts, including_permanent=False)


def run_loss_with_ablation(model: HookedTransformer, tokens, heads_to_ablate, random_cache):
    """Run the model with `heads_to_ablate` mean-ablated and return per-token loss."""
    temp_hooks = [
        (
            tl_utils.get_act_name("v", layer),
            partial(
                head_ablation_hook,
                head_index_to_ablate=head,
                act_name=tl_utils.get_act_name("v", layer),
                random_cache=random_cache,
            ),
        )
        for layer, head in heads_to_ablate
    ]
    return generate_loss_with_hooks(model, tokens, fwd_hooks=temp_hooks)
