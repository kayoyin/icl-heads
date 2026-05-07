import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import os
import random
from typing import *

MODEL_NAME_DICT = {
    '70m': 'EleutherAI/pythia-70m-deduped',
    '160m': 'EleutherAI/pythia-160m-deduped',
    '410m': 'EleutherAI/pythia-410m-deduped',
    '1b': 'EleutherAI/pythia-1b-deduped',
    '1.4b': 'EleutherAI/pythia-1.4b-deduped',
    '2.8b': 'EleutherAI/pythia-2.8b-deduped',
    '6.9b': 'EleutherAI/pythia-6.9b-deduped',
    'gpt2': 'gpt2',
    'gpt2-medium': 'gpt2-medium',
    'gpt2-large': 'gpt2-large',
    'gpt2-xl': 'gpt2-xl',
    '7b': 'meta-llama/Llama-2-7b-hf',
    '8b': 'meta-llama/Meta-Llama-3-8B',
    'qwen': 'Qwen/Qwen2.5-7B',
}

def load_gpt_model_and_tokenizer(model_name:str, device='cuda', ckpt=None):
    """
    Loads a huggingface model and its tokenizer

    Parameters:
    model_name: huggingface name of the model to load (e.g. GPTJ: "EleutherAI/gpt-j-6B", or "EleutherAI/gpt-j-6b")
    device: 'cuda' or 'cpu'
    
    Returns:
    model: huggingface model
    tokenizer: huggingface tokenizer
    MODEL_CONFIG: config variables w/ standardized names
    
    """
    assert model_name is not None

    print("Loading: ", model_name)

    if 'gpt2' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)]}
        
    elif 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.out_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)]}

    elif 'gpt-neox' in model_name.lower() or 'pythia' in model_name.lower():
        if ckpt is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        else:
            cache_dir = f"./{model_name.split('/')[-1]}/step{ckpt}"
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=f"step{ckpt}", cache_dir=cache_dir)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=f"step{ckpt}", cache_dir=cache_dir, torch_dtype=torch.float16).to(device)

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim": model.config.hidden_size,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'gpt_neox.layers.{layer}.attention.dense' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'gpt_neox.layers.{layer}' for layer in range(model.config.num_hidden_layers)]}
        
    elif 'llama' in model_name.lower():
        if '70b' in model_name.lower():
            # use quantization. requires `bitsandbytes` library
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config
            )
        elif 'llama-3' in model_name.lower():
            model_dtype = torch.float16
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)
        else:
            model_dtype = torch.float16
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim":model.config.hidden_size,
                      "name_or_path":model.config._name_or_path,
                      "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)]}
    else:
        raise NotImplementedError("Still working to get this model available!")

    
    return model, tokenizer, MODEL_CONFIG

def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)