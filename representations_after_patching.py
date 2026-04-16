### IMPORTS ### 


import data_utils
import os
import numpy as np
import torch
import tqdm
import pathlib
import argparse
import inflect
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import data_utils
import os
import numpy as np
import torch
import tqdm
import pathlib
import argparse
import inflect
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt 
import seaborn as sns
import gensim
import nltk
import textdistance
from nltk.stem import WordNetLemmatizer
from scipy.stats import pearsonr, kendalltau, spearmanr
from itertools import permutations, product, combinations
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_ind, ttest_rel
from scipy.spatial.distance import cosine
from IPython.display import display, Markdown, Latex
import time
from scipy.special import kl_div
from scipy.stats import binomtest
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import statsmodels.api as sm
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import pandas as pd
import os
import model_utils
import data_utils
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch 
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, XLNetModel, XLMModel, RobertaConfig, BertConfig
from transformers import AutoModelForMaskedLM
import torch
from transformers import AutoModelForSequenceClassification
import numpy as np
import json
import argparse
import json
import os

access_token = os.environ.get('HF_TOKEN_LLAMA')

if access_token is None:
    raise ValueError("HF_TOKEN_LLAMA is not set")
### ARGUMENTS ###

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for extracting representations')
parser.add_argument('--models', nargs='+', type=str, default=None, help='Models to use ("roberta-base", "xlnet-base-cased", and/or "xlm-mlm-xnli15-1024")')
parser.add_argument('--layers', nargs='+', type=int, default=None, help='Which layers to use. Defaults to all excluding embedding. First layer indexed by 1')
parser.add_argument('--device', type=str, default="cpu", help='"cuda" or "cpu"')
parser.add_argument('--representations', nargs='+', type=str, default=None, help='Representations to extract ("mean_pooled", "idiom_sentence")')
parser.add_argument('--rep_loc', type=str, default="./data", help='Where to save representations')
parser.add_argument('--save_attention', dest='save_attention', action='store_true', default=False)
parser.add_argument('--load_if_available', dest='load_if_available', action='store_true', default=False)
parser.add_argument('--amount_of_dataset', type=float, default=1.0, help='Proportion of dataset to use')




def get_final_word_token_from_layers(
    model_name,
    model,
    tokeniser,
    input_ids,
    attention_mask,
    corrected_form_compounds_per_sentence_and,
    layers,
    load_if_available=True,
    batch_size=1,
    rep_type="final_word_standard_attention_head_masked_significant_168",
    rep_loc='./data',
    torch_device="cuda",
    save_attention=False,
    attention_heads_json_path=None,
    attention_heads_by_layer=None,
    mlp_components_json_path=None,
    mlp_components_by_layer=None,
):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=final_word_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence_and": corrected_form_compounds_per_sentence_and}, save_attention=save_attention, attention_heads_json_path=attention_heads_json_path, attention_heads_by_layer=attention_heads_by_layer, mlp_components_json_path=mlp_components_json_path, mlp_components_by_layer=mlp_components_by_layer)


def get_correct_form(word, sentence, stemmer):

    try:
        correct_form = word_tokenize(sentence)[np.where(np.array([stemmer.stem(x) for x in word_tokenize(sentence)]) == stemmer.stem(word))[0][0]]
    except IndexError:
        return ''

    return correct_form

def get_corrected_form_compounds_per_sentence(sentences, mod_head_words_per_sentence):
    # 
    stemmer = PorterStemmer()

    # Get first matching stem
    correct_mod_words = [get_correct_form(x[0], sentences[i], stemmer) for i, x in enumerate(mod_head_words_per_sentence)]
    correct_head_nouns = [get_correct_form(x[1], sentences[i], stemmer) for i, x in enumerate(mod_head_words_per_sentence)]

    res = list(zip(sentences, mod_head_words_per_sentence[:, 0], correct_mod_words, mod_head_words_per_sentence[:, 1], correct_head_nouns))

def get_corrected_form_compounds_per_sentence_with_context(sentences, mod_head_words_per_sentence_with_context):
    # 
    stemmer = PorterStemmer()

    # Get first matching stem
    correct_mod_words = [get_correct_form(x[0], sentences[i], stemmer) for i, x in enumerate(mod_head_words_per_sentence_with_context)]
    correct_head_nouns = [get_correct_form(x[1], sentences[i], stemmer) for i, x in enumerate(mod_head_words_per_sentence_with_context)]
    
    res = list(zip(sentences, mod_head_words_per_sentence_with_context[:, 0], correct_mod_words, mod_head_words_per_sentence_with_context[:, 1], correct_head_nouns))



def get_corrected_form_compounds_per_sentence_with_and(sentences, and_head_words_per_sentence):
    # 
    stemmer = PorterStemmer()

    # Get first matching stem
    correct_and_words = [get_correct_form(x[1], sentences[i], stemmer) for i, x in enumerate(and_head_words_per_sentence)]
    correct_head_nouns = [get_correct_form(x[0], sentences[i], stemmer) for i, x in enumerate(and_head_words_per_sentence)]
    
    res = list(zip(sentences, and_head_words_per_sentence[:, 0], correct_head_nouns, and_head_words_per_sentence[:, 1], correct_and_words))


def get_corrected_form_compounds_per_sentence_with_context_and(sentences, and_head_words_per_sentence_with_context):
    # 
    stemmer = PorterStemmer()

    # Get first matching stem
    correct_and_words = [get_correct_form(x[1], sentences[i], stemmer) for i, x in enumerate(and_head_words_per_sentence_with_context)]
    correct_head_nouns = [get_correct_form(x[0], sentences[i], stemmer) for i, x in enumerate(and_head_words_per_sentence_with_context)]
    
    res = list(zip(sentences, and_head_words_per_sentence_with_context[:, 0], correct_head_nouns, and_head_words_per_sentence_with_context[:, 1], correct_and_words))

 


def search_sequence_numpy(arr,seq):
    # https://stackoverflow.com/a/36535397
    # 
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []   





def final_word_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence_and):

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence_and])[i:i+batch_size]
    

    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)

    final_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    final_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in final_word_input_ids_per_sent_raw]
    final_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(final_word_input_ids_per_sent)]
    final_word_reps = np.vstack([reps[final_word_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    return final_word_reps


def normalize_attention_heads_by_layer(raw):
    """
    Build mapping layer -> list of head indices to zero.

    Expects layer indices to be the same convention as the model module list:
    0-based layer indices (i.e. JSON layer 0 targets `model.model.layers[0]`).

    Accepts:
      - list of [layer, head] or (layer, head)
      - dict: layer (str or int) -> list of head ints, or list of [*, head] pairs
    """
    if isinstance(raw, str):
        with open(raw, "r") as f:
            raw = json.load(f)
    layer_to_heads = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            layer = int(k)
            heads = []
            for item in v:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    heads.append(int(item[1]))
                else:
                    heads.append(int(item))
            layer_to_heads[layer] = heads
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                raise ValueError(
                    "Each attention head entry must be [layer, head]; got %r" % (item,)
                )
            layer, head = int(item[0]), int(item[1])
            layer_to_heads.setdefault(layer, []).append(head)
    else:
        raise TypeError("attention heads JSON must be dict or list, got %s" % type(raw))
    return layer_to_heads


def normalize_mlp_components_by_layer(raw):
    """
    Build mapping layer -> list of MLP component indices to zero.

    Accepts:
      - list of [layer, component] or (layer, component)
      - dict: layer (str or int) -> list of component ints, or list of [*, component] pairs
      - str path to JSON file in either format above
    """
    if isinstance(raw, str):
        with open(raw, "r") as f:
            raw = json.load(f)

    layer_to_components = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            layer = int(k)
            comps = []
            for item in v:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    comps.append(int(item[1]))
                else:
                    comps.append(int(item))
            layer_to_components[layer] = comps
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                raise ValueError(
                    "Each MLP component entry must be [layer, component]; got %r" % (item,)
                )
            layer, component = int(item[0]), int(item[1])
            layer_to_components.setdefault(layer, []).append(component)
    else:
        raise TypeError("MLP components JSON must be dict or list, got %s" % type(raw))

    return layer_to_components



def _make_o_proj_zero_heads_hook(head_indices, num_heads, head_dim):
    """Forward hook on Llama (and similar) o_proj: zeros selected output heads."""
    heads_set = {int(h) for h in head_indices}

    def hook(module, inp, output):
        if output is None or output.ndim != 3:
            return output
        b, s, d = output.shape
        if d != num_heads * head_dim:
            return output
        view = output.view(b, s, num_heads, head_dim)
        for h in heads_set:
            if 0 <= h < num_heads:
                view[:, :, h, :].zero_()
        return output

    return hook


def _make_c_proj_zero_heads_pre_hook(head_indices, num_heads, head_dim):
    """Pre-hook on GPT c_proj: zeros selected heads in c_proj input."""
    heads_set = {int(h) for h in head_indices}

    def pre_hook(module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if x is None or x.ndim != 3:
            return inputs
        b, s, d = x.shape
        if d != num_heads * head_dim:
            return inputs
        view = x.view(b, s, num_heads, head_dim)
        for h in heads_set:
            if 0 <= h < num_heads:
                view[:, :, h, :].zero_()
        return (x,) + tuple(inputs[1:])

    return pre_hook


def _make_mlp_input_zero_components_pre_hook(component_indices):
    """Pre-hook: zeros selected feature components in MLP projection input."""
    components_set = {int(c) for c in component_indices}

    def pre_hook(module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if x is None or x.ndim != 3:
            return inputs
        _, _, d = x.shape
        for comp in components_set:
            if 0 <= comp < d:
                x[:, :, comp].zero_()
        return (x,) + tuple(inputs[1:])

    return pre_hook


def _register_attention_head_o_proj_hooks(model, layer_to_heads):
    """
    Register hooks on model.model.layers[*].self_attn.o_proj.
    layer_to_heads: 0-based layer index -> list of head indices (0-based).
    Returns list of RemovableHandle to remove after the forward pass.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Expected a CausalLM with model.model.layers (e.g. Llama).")
    cfg = model.config
    default_head_dim = getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
    )
    handles = []
    # Ensure deterministic intervention order by registering hooks
    # from lowest -> highest layer index.
    for layer_1based in sorted(layer_to_heads.keys(), key=lambda x: int(x)):
        heads = layer_to_heads[layer_1based]
        if not heads:
            continue
        layer_0 = int(layer_1based)
        if layer_0 < 0 or layer_0 >= len(model.model.layers):
            continue
        attn = model.model.layers[layer_0].self_attn
        if not hasattr(attn, "o_proj"):
            raise AttributeError(
                "Layer %d self_attn has no o_proj; cannot apply head masking." % layer_1based
            )
        o_proj = attn.o_proj
        out_features = o_proj.out_features
        if out_features % default_head_dim != 0:
            head_dim = cfg.hidden_size // cfg.num_attention_heads
        else:
            head_dim = default_head_dim
        n_heads = out_features // head_dim
        h = o_proj.register_forward_hook(
            _make_o_proj_zero_heads_hook(sorted(map(int, heads)), n_heads, head_dim)
        )
        handles.append(h)
    return handles


def _register_attention_head_c_proj_hooks_gpt(model, layer_to_heads):
    """
    Register pre-hooks on model.transformer.h[*].attn.c_proj.
    layer_to_heads: 0-based layer index -> list of head indices (0-based).
    Returns list of RemovableHandle to remove after the forward pass.
    """
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise AttributeError("Expected a GPT-style model with transformer.h layers.")

    cfg = model.config
    num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    if num_heads is None or hidden_size is None:
        raise AttributeError("Could not infer GPT num heads / hidden size from config.")
    head_dim = hidden_size // num_heads

    handles = []
    for layer_idx in sorted(layer_to_heads.keys(), key=lambda x: int(x)):
        heads = layer_to_heads[layer_idx]
        if not heads:
            continue
        layer_0 = int(layer_idx)
        if layer_0 < 0 or layer_0 >= len(model.transformer.h):
            continue
        attn = model.transformer.h[layer_0].attn
        if not hasattr(attn, "c_proj"):
            raise AttributeError(
                "Layer %d attn has no c_proj; cannot apply head masking." % layer_idx
            )
        c_proj = attn.c_proj
        h = c_proj.register_forward_pre_hook(
            _make_c_proj_zero_heads_pre_hook(sorted(map(int, heads)), num_heads, head_dim)
        )
        handles.append(h)
    return handles


def _register_mlp_down_proj_input_hooks(model, layer_to_components):
    """
    Register pre-hooks on model.model.layers[*].mlp.down_proj.
    layer_to_components: 0-based layer index -> list of component indices (0-based).
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Expected a CausalLM with model.model.layers (e.g. Llama).")

    handles = []
    for layer_idx in sorted(layer_to_components.keys(), key=lambda x: int(x)):
        components = layer_to_components[layer_idx]
        if not components:
            continue
        layer_0 = int(layer_idx)
        if layer_0 < 0 or layer_0 >= len(model.model.layers):
            continue
        mlp = model.model.layers[layer_0].mlp
        if not hasattr(mlp, "down_proj"):
            raise AttributeError(
                "Layer %d mlp has no down_proj; cannot apply MLP masking." % layer_idx
            )
        h = mlp.down_proj.register_forward_pre_hook(
            _make_mlp_input_zero_components_pre_hook(sorted(map(int, components)))
        )
        handles.append(h)
    return handles


def _register_mlp_c_proj_input_hooks_gpt(model, layer_to_components):
    """
    Register pre-hooks on model.transformer.h[*].mlp.c_proj.
    layer_to_components: 0-based layer index -> list of component indices (0-based).
    """
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise AttributeError("Expected a GPT-style model with transformer.h layers.")

    handles = []
    for layer_idx in sorted(layer_to_components.keys(), key=lambda x: int(x)):
        components = layer_to_components[layer_idx]
        if not components:
            continue
        layer_0 = int(layer_idx)
        if layer_0 < 0 or layer_0 >= len(model.transformer.h):
            continue
        mlp = model.transformer.h[layer_0].mlp
        if not hasattr(mlp, "c_proj"):
            raise AttributeError(
                "Layer %d mlp has no c_proj; cannot apply MLP masking." % layer_idx
            )
        h = mlp.c_proj.register_forward_pre_hook(
            _make_mlp_input_zero_components_pre_hook(sorted(map(int, components)))
        )
        handles.append(h)
    return handles






def get_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers,
    token_selector=final_word_selector, load_if_available=True, batch_size=1, rep_type="sentence_pair_cls", torch_device="cuda", save_reps=True,
    data_loc='./data', add_arg_dict={}, middle_dim=None, save_attention=False,
    attention_heads_json_path=None, attention_heads_by_layer=None,
    mlp_components_json_path=None, mlp_components_by_layer=None):
    """
    Extracts token representations from specified layers of a model and optionally saves them.
    Optional attention-head and MLP-component intervention.
    - attention_heads_*: zero selected attention heads
    - mlp_components_*: zero selected MLP input components
    Both can be enabled at once and will run in the same forward pass.
    """
    # Get file locations for layer representations
    rep_locs_per_layer = [
        data_utils.get_hidden_state_file(model_name, layer=x, rep_type=rep_type, data_loc=data_loc)
        for x in layers
    ]
    load_reps = load_if_available and all(os.path.isfile(x) for x in rep_locs_per_layer)

    if load_reps:
        return [np.load(x) for x in rep_locs_per_layer]

    print(f'Extracting representations from model for layers {layers}')
    
    # Move data to the appropriate device
    input_ids = input_ids.to(torch_device)
    attention_mask = attention_mask.to(torch_device)
    model.to(torch_device)

    attn_hook_handles = []
    mlp_hook_handles = []

    heads_by_layer = None
    if attention_heads_by_layer is not None:
        heads_by_layer = attention_heads_by_layer
    elif attention_heads_json_path:
        heads_by_layer = normalize_attention_heads_by_layer(attention_heads_json_path)

    if heads_by_layer:
        model_name_lower = model_name.lower()
        if "llama" in model_name_lower:
            attn_hook_handles = _register_attention_head_o_proj_hooks(model, heads_by_layer)
            hook_target = "o_proj"
        elif "gpt" in model_name_lower:
            attn_hook_handles = _register_attention_head_c_proj_hooks_gpt(model, heads_by_layer)
            hook_target = "c_proj"
        else:
            # Default to llama-style hook if model family string is not explicit.
            attn_hook_handles = _register_attention_head_o_proj_hooks(model, heads_by_layer)
            hook_target = "o_proj"
        print(
            "Attention-head intervention (zero via %s): layers %s"
            % (hook_target, {k: v for k, v in heads_by_layer.items() if v})
        )

    mlp_by_layer = None
    if mlp_components_by_layer is not None:
        mlp_by_layer = mlp_components_by_layer
    elif mlp_components_json_path:
        mlp_by_layer = normalize_mlp_components_by_layer(mlp_components_json_path)

    if mlp_by_layer:
        model_name_lower = model_name.lower()
        if "llama" in model_name_lower:
            mlp_hook_handles = _register_mlp_down_proj_input_hooks(model, mlp_by_layer)
            mlp_hook_target = "down_proj"
        elif "gpt" in model_name_lower:
            mlp_hook_handles = _register_mlp_c_proj_input_hooks_gpt(model, mlp_by_layer)
            mlp_hook_target = "c_proj"
        else:
            mlp_hook_handles = _register_mlp_down_proj_input_hooks(model, mlp_by_layer)
            mlp_hook_target = "down_proj"
        print(
            "MLP intervention (zero via %s input): layers %s"
            % (mlp_hook_target, {k: v for k, v in mlp_by_layer.items() if v})
        )

    # Initialize token representations dynamically (add 2 for idiom representations)
    tokens_per_layer = [
        np.zeros((input_ids.shape[0], model.config.hidden_size))
        for _ in layers
    ]
    
    # Optionally initialize attention representations
    if save_attention:
        seq_len = max(len(tokeniser.decode(x)) for x in input_ids)
        attention_per_layer = [-np.ones((input_ids.shape[0], seq_len**2)) for _ in layers]

    try:
        # Extract representations
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
                outputs = model(
                    input_ids[i:i+batch_size],
                    attention_mask=attention_mask[i:i+batch_size],
                    output_attentions=save_attention,
                    output_hidden_states=True
                )
                # the hidden states are the vector representations of the tokens in the sentence
                # the first hidden state is the embeddings
                # these vectors are transformed after each layer of the model

                hidden_states = outputs.hidden_states[1:]
                add_arg_dict["i"] = i

                # Process each layer
                for idx, layer in enumerate(layers):
                    tokens_per_layer[idx][i:i+batch_size] = token_selector(
                        model, model_name, tokeniser, hidden_states,
                        input_ids[i:i+batch_size], layer, batch_size, **add_arg_dict
                    )
    finally:
        for h in attn_hook_handles:
            h.remove()
        for h in mlp_hook_handles:
            h.remove()

    # Save representations
    if save_reps:
        for idx, layer in enumerate(layers):
            layer_file = data_utils.get_hidden_state_file(
                model_name, layer=layer, rep_type=rep_type, data_loc=data_loc
            )
            pathlib.Path('/'.join(layer_file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
            np.save(layer_file, tokens_per_layer[idx])

    if save_attention:
        for idx, layer in enumerate(layers):
            atten_file = data_utils.get_hidden_state_file(
                model_name, layer=layer, rep_type=f"{rep_type}_attention", data_loc=data_loc
            )
            pathlib.Path('/'.join(atten_file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
            np.save(atten_file, attention_per_layer[idx])

    return tokens_per_layer


def load_significant_neurons(
    path="top_67_heads_llama3b.json",
):
    """Load JSON of (layer, head) pairs into {layer: [heads,...]} (0-based layers/heads)."""
    with open(path, "r") as f:
        top_heads = json.load(f)
    if isinstance(top_heads, dict):
        # e.g. {"13": [[0, 5], [0, 7]]} or {"13": [5, 7]}
        return normalize_attention_heads_by_layer(top_heads)
    # e.g. [[13, 5], [13, 7], ...] or legacy iterable of (layer, heads_list)
    if top_heads and isinstance(top_heads[0], (list, tuple)) and len(top_heads[0]) == 2:
        second = top_heads[0][1]
        if isinstance(second, (list, tuple)):
            return {
                int(layer): [int(h[1]) for h in heads]
                for layer, heads in top_heads
            }
    return normalize_attention_heads_by_layer(top_heads)

def load_significant_mlp_neurons(path= "top_67_mlp_components_llama3b.json"):
    """Load JSON of (layer, component) pairs into {layer: [components,...]}."""
    with open(path, "r") as f:
        top_mlp_neurons = json.load(f)
    return normalize_mlp_components_by_layer(top_mlp_neurons)



def load_significant_heads_pairs(path):
    """
    Load significant heads JSON into a flat list of (layer, head) pairs.

    Expected JSON formats:
      - [[layer, head], [layer, head], ...]  (0-based for both)
      - { "layer": [head, head, ...] }      (0-based for both)
    """
    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        # [[layer, head], ...]
        if raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2:
            return [(int(l), int(h)) for l, h in raw]
        # if the JSON is something else, fall back to normalization + flatten
        by_layer = normalize_attention_heads_by_layer(raw)
        return [(int(layer), int(head)) for layer, heads in by_layer.items() for head in heads]

    if isinstance(raw, dict):
        by_layer = normalize_attention_heads_by_layer(raw)
        return [(int(layer), int(head)) for layer, heads in by_layer.items() for head in heads]

    raise TypeError(f"Unsupported significant heads JSON type: {type(raw)}")


def load_significant_mlp_pairs(path):
    """
    Load significant MLP JSON into a flat list of (layer, component) pairs.

    Expected JSON formats:
      - [[layer, component], [layer, component], ...]  (0-based for both)
      - { "layer": [component, component, ...] }       (0-based for both)
    """
    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        # [[layer, component], ...]
        if raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2:
            return [(int(l), int(c)) for l, c in raw]
        by_layer = normalize_mlp_components_by_layer(raw)
        return [(int(layer), int(comp)) for layer, comps in by_layer.items() for comp in comps]

    if isinstance(raw, dict):
        by_layer = normalize_mlp_components_by_layer(raw)
        return [(int(layer), int(comp)) for layer, comps in by_layer.items() for comp in comps]

    raise TypeError(f"Unsupported significant MLP JSON type: {type(raw)}")


def generate_random_heads_by_layer_from_pairs(significant_pairs, num_heads, seed=None):
    """
    For each layer mentioned in `significant_pairs`, sample the same *count* of heads
    (no replacement) from [0, num_heads).

    Returns: {layer: [heads,...]} suitable for `attention_heads_by_layer=...`.
    """
    rng = np.random.default_rng(seed)

    layer_to_heads = {}
    for layer, head in significant_pairs:
        layer_to_heads.setdefault(int(layer), set()).add(int(head))

    random_pairs = []
    for layer in sorted(layer_to_heads.keys(), key=lambda x: int(x)):
        k = len(layer_to_heads[layer])
        # If a layer requests more heads than exist, cap at num_heads.
        k = min(k, int(num_heads))
        sampled = rng.choice(int(num_heads), size=k, replace=False).tolist()
        random_pairs.extend([(int(layer), int(h)) for h in sampled])

    return normalize_attention_heads_by_layer(random_pairs)


def _infer_mlp_component_dim(model):
    """
    Infer number of available MLP components for sampling.
    For LLaMA this is `down_proj.in_features`; for GPT this is `c_proj.in_features`.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        first_mlp = model.model.layers[0].mlp
        if hasattr(first_mlp, "down_proj") and hasattr(first_mlp.down_proj, "in_features"):
            return int(first_mlp.down_proj.in_features)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h") and len(model.transformer.h) > 0:
        first_mlp = model.transformer.h[0].mlp
        if hasattr(first_mlp, "c_proj") and hasattr(first_mlp.c_proj, "in_features"):
            return int(first_mlp.c_proj.in_features)
    raise AttributeError("Could not infer MLP component dimension from model.")


def generate_random_mlp_components_by_layer_from_pairs(significant_pairs, num_components, seed=None):
    """
    For each layer mentioned in `significant_pairs`, sample the same *count* of MLP
    components (no replacement) from [0, num_components).

    Returns: {layer: [components,...]} suitable for `mlp_components_by_layer=...`.
    """
    rng = np.random.default_rng(seed)

    layer_to_components = {}
    for layer, component in significant_pairs:
        layer_to_components.setdefault(int(layer), set()).add(int(component))

    random_pairs = []
    for layer in sorted(layer_to_components.keys(), key=lambda x: int(x)):
        k = len(layer_to_components[layer])
        k = min(k, int(num_components))
        sampled = rng.choice(int(num_components), size=k, replace=False).tolist()
        random_pairs.extend([(int(layer), int(c)) for c in sampled])

    return normalize_mlp_components_by_layer(random_pairs)


def generate_random_heads_by_layer_from_significant_file(significant_path, model, seed=None):
    """
    Convenience wrapper: read significant heads JSON and return random heads by layer.
    """
    significant_pairs = load_significant_heads_pairs(significant_path)
    num_heads = model.config.num_attention_heads
    return generate_random_heads_by_layer_from_pairs(
        significant_pairs=significant_pairs,
        num_heads=num_heads,
        seed=seed,
    )


def generate_random_mlp_components_by_layer_from_significant_file(significant_path, model, seed=None):
    """
    Convenience wrapper: read significant MLP JSON and return random components by layer.
    """
    significant_pairs = load_significant_mlp_pairs(significant_path)
    num_components = _infer_mlp_component_dim(model)
    return generate_random_mlp_components_by_layer_from_pairs(
        significant_pairs=significant_pairs,
        num_components=num_components,
        seed=seed,
    )




def attention_head_masking():
    model_name = "meta-llama/Llama-3.2-3B"
    batch_size = 1
    layers = list(range(1, 29))
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    rep_loc = "./data"
    
    print(f"Model: {model_name}")
    print(f"Device: {torch_device}")
    print(f"Layers to extract from: {layers}")

    model, tokeniser = model_utils.load_model(model_name)
    if tokeniser.pad_token is None:
        if tokeniser.eos_token:
            tokeniser.pad_token = tokeniser.eos_token
        else:
            tokeniser.add_special_tokens({'pad_token': '<pad>'})
    
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")

    sentences = data_utils.get_no_context_sentences()
    corrected_form_compounds_per_sentence_and = data_utils.load_correct_form_no_context_and()
    
    inputs = tokeniser(sentences.tolist(), max_length=512, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Move data to device
    input_ids = input_ids.to(torch_device)
    attention_mask = attention_mask.to(torch_device)
    model.to(torch_device)

    top_heads_dict = load_significant_neurons()
    top_mlp_dict = None
    # top_heads_dict = generate_random_heads_by_layer_from_significant_file("top_34_heads_llama3b.json", model, seed=42)

    rep_type = f'final_word_literal_attention_head_masked_significant_67'

    get_final_word_token_from_layers(
        model_name,
        model,
        tokeniser,
        input_ids,
        attention_mask,
        corrected_form_compounds_per_sentence_and,
        layers=layers,
        load_if_available=False,
        batch_size=batch_size,
        rep_type=rep_type,
        rep_loc=rep_loc,
        torch_device=torch_device,
        save_attention=False,
        attention_heads_by_layer=top_heads_dict,
        mlp_components_by_layer=top_mlp_dict,
    )



def random_attention_head_masking():
    model_name = "meta-llama/Llama-3.2-3B"
    batch_size = 1
    layers = list(range(1, 29))
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    rep_loc = "./data"
    
    print(f"Model: {model_name}")
    print(f"Device: {torch_device}")
    print(f"Layers to extract from: {layers}")

    model, tokeniser = model_utils.load_model(model_name)
    if tokeniser.pad_token is None:
        if tokeniser.eos_token:
            tokeniser.pad_token = tokeniser.eos_token
        else:
            tokeniser.add_special_tokens({'pad_token': '<pad>'})
    
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")

    sentences = data_utils.get_context_sentences()
    corrected_form_compounds_per_sentence_and = data_utils.load_correct_form_context_and()
    
    inputs = tokeniser(sentences.tolist(), max_length=512, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Move data to device
    input_ids = input_ids.to(torch_device)
    attention_mask = attention_mask.to(torch_device)
    model.to(torch_device)

    significant_path = "top_168_heads_llama3b.json"
    num_random_runs = 5
    base_random_seed = 10042

    for run_idx in range(num_random_runs):
        seed = base_random_seed + run_idx
        top_heads_dict = generate_random_heads_by_layer_from_significant_file(
            significant_path, model, seed=seed
        )
        top_mlp_dict = None
        rep_type = f"final_word_context_attention_head_masked_168_random_run{run_idx + 1}"

        print(
            "Random head masking run %d/%d (seed=%s); layers -> head counts: %s"
            % (
                run_idx + 1,
                num_random_runs,
                seed,
                {k: len(v) for k, v in sorted(top_heads_dict.items())},
            )
        )

        get_final_word_token_from_layers(
            model_name,
            model,
            tokeniser,
            input_ids,
            attention_mask,
            corrected_form_compounds_per_sentence_and,
            layers=layers,
            load_if_available=False,
            batch_size=batch_size,
            rep_type=rep_type,
            rep_loc=rep_loc,
            torch_device=torch_device,
            save_attention=False,
            attention_heads_by_layer=top_heads_dict,
            mlp_components_by_layer=top_mlp_dict,
        )


def mlp_attention_masking():
    """ runs both attention head masking and mlp component masking """
    model_name = "meta-llama/Llama-3.2-3B"
    batch_size = 1
    layers = list(range(1, 29))
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    rep_loc = "./data"
    
    print(f"Model: {model_name}")
    print(f"Device: {torch_device}")
    print(f"Layers to extract from: {layers}")

    model, tokeniser = model_utils.load_model(model_name)
    if tokeniser.pad_token is None:
        if tokeniser.eos_token:
            tokeniser.pad_token = tokeniser.eos_token
        else:
            tokeniser.add_special_tokens({'pad_token': '<pad>'})
    
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")

    sentences = data_utils.get_no_context_sentences()
    corrected_form_compounds_per_sentence_and = data_utils.load_correct_form_no_context_and()
    
    inputs = tokeniser(sentences.tolist(), max_length=512, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Move data to device
    input_ids = input_ids.to(torch_device)
    attention_mask = attention_mask.to(torch_device)
    model.to(torch_device)

    top_heads_dict = load_significant_neurons()
    top_mlp_dict = load_significant_mlp_neurons()

    rep_type = f"final_literal_attention_head_masked_67_mlp_masked_67"

    get_final_word_token_from_layers(
        model_name,
        model,
        tokeniser,
        input_ids,
        attention_mask,
        corrected_form_compounds_per_sentence_and,
        layers=layers,
        load_if_available=False,
        batch_size=batch_size,
        rep_type=rep_type,
        rep_loc=rep_loc,
        torch_device=torch_device,
        save_attention=False,
        attention_heads_by_layer=top_heads_dict,
        mlp_components_by_layer=top_mlp_dict,
    )


def random_mlp_attention_masking():
    """Run 5 random joint interventions over attention heads + MLP components."""
    model_name = "meta-llama/Llama-3.2-3B"
    batch_size = 1
    layers = list(range(1, 29))
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    rep_loc = "./data"

    print(f"Model: {model_name}")
    print(f"Device: {torch_device}")
    print(f"Layers to extract from: {layers}")

    model, tokeniser = model_utils.load_model(model_name)
    if tokeniser.pad_token is None:
        if tokeniser.eos_token:
            tokeniser.pad_token = tokeniser.eos_token
        else:
            tokeniser.add_special_tokens({'pad_token': '<pad>'})

    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")

    sentences = data_utils.get_standard_sentences()
    corrected_form_compounds_per_sentence_and = data_utils.load_correct_form_standard_and()

    inputs = tokeniser(sentences.tolist(), max_length=512, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Move data to device
    input_ids = input_ids.to(torch_device)
    attention_mask = attention_mask.to(torch_device)
    model.to(torch_device)

    significant_heads_path = "top_67_heads_llama3b.json"
    significant_mlp_path = "top_67_mlp_components_llama3b.json"
    num_random_runs = 5
    base_random_seed = 12042

    for run_idx in range(num_random_runs):
        seed = base_random_seed + run_idx
        top_heads_dict = generate_random_heads_by_layer_from_significant_file(
            significant_heads_path, model, seed=seed
        )
        top_mlp_dict = generate_random_mlp_components_by_layer_from_significant_file(
            significant_mlp_path, model, seed=seed
        )
        rep_type = f"final_standard_attention_head_masked_67_mlp_masked_67_random_run{run_idx + 1}"

        print(
            "Random joint masking run %d/%d (seed=%s); head counts=%s; mlp counts=%s"
            % (
                run_idx + 1,
                num_random_runs,
                seed,
                {k: len(v) for k, v in sorted(top_heads_dict.items())},
                {k: len(v) for k, v in sorted(top_mlp_dict.items())},
            )
        )

        get_final_word_token_from_layers(
            model_name,
            model,
            tokeniser,
            input_ids,
            attention_mask,
            corrected_form_compounds_per_sentence_and,
            layers=layers,
            load_if_available=False,
            batch_size=batch_size,
            rep_type=rep_type,
            rep_loc=rep_loc,
            torch_device=torch_device,
            save_attention=False,
            attention_heads_by_layer=top_heads_dict,
            mlp_components_by_layer=top_mlp_dict,
        )

if __name__ == "__main__":
    random_mlp_attention_masking()
