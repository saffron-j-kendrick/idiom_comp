### IMPORTS

from IPython.display import clear_output
import nnsight
from nnsight import CONFIG
from nnsight import LanguageModel, util
from nnsight.tracing.graph.proxy import Proxy
import plotly.express as px
import plotly.io as pio
import numpy as np
import torch
import kaleido
import matplotlib.pyplot as plt
import seaborn as sns
import einops
import pickle
import cv2
import pandas as pd
import argparse
import json
import os
import torch.nn.functional as F



### PARSER ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2", help="Model to use: gpt2, llama3b")
parser.add_argument("--intervention", type=str, default="residual_stream", help="Intervention to use: residual_stream, attention_heads, mlp")
parser.add_argument("--dataset", type=str, default="combined_dataset.json", help="Dataset to use: combined_dataset.json")
parser.add_argument("--averaging", type=bool, default=False, help="Whether to average the intervention results over the dataset")
parser.add_argument("--ablation", type=bool, default=False, help="Whether to run the ablation study")
parser.add_argument("--random_ablation", type=bool, default=False, help="Whether to run the random ablation study")
parser.add_argument("--top_k", type=int, default=10, help="Number of top heads to patch/zero")
parser.add_argument("--clean_run_ablation", type=bool, default=False, help="Whether to run the clean run ablation study")
parser.add_argument("--visualise_attention_head_patterns", type=bool, default=False, help="Whether to visualise the attention head patterns")

### FUNCTIONS

def plot_ioi_patching_results(model,
                              ioi_patching_results,
                              x_labels,
                              prompt_idiom,
                              plot_title="Normalized Logit Difference"):
    N_LAYERS = len(model.transformer.h)
    # 1. Extract values from nnsight Proxies
    # Manually extract values since Proxies are evaluated after trace context
    unwrapped_results = []
    for layer_results in ioi_patching_results:
        layer_values = []
        for result in layer_results:
            # Check if it's already a float/int or if it's a Proxy
            if isinstance(result, Proxy):
                layer_values.append(result.value)
            else:
                layer_values.append(result)
        unwrapped_results.append(layer_values)

    # 2. Convert to a 2D numpy array for plotting
    data = np.array(unwrapped_results)

    # 3. Create the plot
    y_labels = list(range(1, N_LAYERS + 1))
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels, # Shows layer numbers
        cmap="RdBu",
        center=0.0,
        cbar_kws={'label': 'Norm. Logit Diff'}
    )

    plt.title(plot_title)
    plt.xlabel("Token Position")
    plt.ylabel("Layer")

    # Rotate x-labels if they are crowded
    plt.xticks(rotation=45, ha='right')

    # Save the file
    plt.tight_layout()
    plt.savefig(f"figures/patching_results_{prompt_idiom}.png")
    plt.savefig(f"figures/patching_results_{prompt_idiom}.eps")
    plt.show()

    return plt.gcf()

def plot_ioi_patching_results_attention(model,
                              ioi_patching_results,
                              x_labels,
                              prompt_idiom,
                              plot_title="Normalized Logit Difference"):

    N_LAYERS = len(model.transformer.h)
    # 1. Extract values from nnsight Proxies
    # Manually extract values since Proxies are evaluated after trace context
    unwrapped_results = []
    for layer_results in ioi_patching_results:
        layer_values = []
        for result in layer_results:
            # Check if it's already a float/int or if it's a Proxy
            if isinstance(result, Proxy):
                layer_values.append(result.value)
            else:
                layer_values.append(result)
        unwrapped_results.append(layer_values)

    # 2. Convert to a 2D numpy array for plotting
    data = np.array(unwrapped_results)

    # 3. Create the plot
    y_labels = list(range(1, N_LAYERS + 1))
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels, # Shows layer numbers
        cmap="RdBu",
        center=0.0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Norm. Logit Diff'}
    )

    plt.title(plot_title)
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")

    # Rotate x-labels if they are crowded
    plt.xticks(rotation=45, ha='right')

    # Save the file
    plt.tight_layout()
    plt.savefig(f"figures/patching_results_{prompt_idiom}_attention.png")
    plt.savefig(f"figures/patching_results_{prompt_idiom}_attention.eps")
    plt.show()

    return plt.gcf()


def plot_kl_diff(kl_diff, N_LAYERS, N_HEADS, k):
    plt.figure(figsize=(10, 6))
    y_labels = list(range(1, N_LAYERS + 1))
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    sns.heatmap(
        kl_diff,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="RdBu",
        cbar_kws={'label': 'KL Divergence'},
        center=0
    )
    plt.xlabel("Token position")
    plt.ylabel("Layer")
    plt.title(f"Top {k} Attention Head Ablation KL Divergence Diff")
    plt.savefig(f"figures/top_{k}_attention_head_ablation_kl_diff_gpt2.png")
    plt.savefig(f"figures/top_{k}_attention_head_ablation_kl_diff_gpt2.eps")
    plt.gca() # Often helpful to have Layer 0 at the bottom
    plt.show()
    return plt.gcf()



def visualize_attention(model, prompt_idiom, layer, head):
    # We need attention weights to be returned by the model.
    # Some nnsight proxy objects can error if you index them while the underlying
    # attention weights are `None`, so we delay indexing until after the trace.
    try:
        trace_ctx = model.trace(prompt_idiom, output_attentions=True)
    except TypeError:
        # Fallback for older nnsight versions that don't accept forward kwargs.
        trace_ctx = model.trace(prompt_idiom)

    # Save the *full* attention output for the layer. We will index into it after the trace.
    with trace_ctx:
        if hasattr(model, "transformer"):  # GPT-2 style
            attn_output_proxy = model.transformer.h[layer].attn.output.save()
        elif hasattr(model, "model"):  # LLaMA style (best-effort)
            attn_output_proxy = model.model.layers[layer].self_attn.output.save()
        else:
            raise RuntimeError("Unsupported model type for attention visualization.")

    attn_output_value = attn_output_proxy.value
    if attn_output_value is None:
        raise RuntimeError(
            "Attention output is None. "
            "Tried to trace with `output_attentions=True`, but the model "
            "didn't produce attention outputs."
        )

    # Convention in many HF attention modules: output tuple like (attn_output, attn_weights, ...)
    # We follow the same assumption as the original implementation.
    try:
        attn_weights_all = attn_output_value[1]
    except Exception as e:
        raise RuntimeError(
            "Couldn't extract attention weights from the traced attention output."
        ) from e

    if attn_weights_all is None:
        raise RuntimeError(
            "Attention weights are None. "
            "Tried to trace with `output_attentions=True`, but the model "
            "still didn't return attention weights."
        )

    # Convert to numpy for plotting: expected [batch, head, query_pos, key_pos]
    weights = attn_weights_all[0, head].detach().cpu().numpy()

    # Tokenize for labels
    tokens = [model.tokenizer.decode(t) for t in model.tokenizer.encode(prompt_idiom)]
   
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        cbar_kws={'label': 'Attention Probability'}
    )
    plt.title(f"Attention Pattern: Layer {layer+1}, Head {head+1}")
    plt.xlabel("Key (Source Token)")
    plt.ylabel("Query (Target Token)")
    plt.xticks(rotation=45)
    plt.savefig(f"figures/attention_head_patterns_{prompt_idiom}_{layer+1}_{head+1}.png")
    plt.savefig(f"figures/attention_head_patterns_{prompt_idiom}_{layer+1}_{head+1}.eps")
    plt.show()
    return plt.gcf()


def residual_stream_patching(N_LAYERS, prompt_idiom, prompt_literal, correct_answer_idx, incorrect_answer_idx, min_token_len):
    """ Replaces the residual stream at the end of each layer with the residual stream from the clean prompt """
    with model.trace(prompt_idiom) as tracer:
        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
        clean_hs = [model.transformer.h[i].output[0].save() for i in range(N_LAYERS)]
        clean_logits = model.lm_head.output.save()
        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
    # Convert Proxy objects to numpy arrays after trace context exits
    
    clean_hs_np = [act.value.detach().cpu().numpy() for act in clean_hs]
    for layer_idx, act in enumerate(clean_hs_np):
        np.save(f"data/clean_hidden_state_{layer_idx}.npy", act)
    
    with model.trace(prompt_literal) as tracer:
        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
        corrupted_logits = model.lm_head.output
        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
        # print(f"corrupt logit diff {corrupted_logit_diff}")

    residual_stream_patching_intervention = []
    for layer_idx in range(N_LAYERS):
        clean_hs_np = np.load(f"data/clean_hidden_state_{layer_idx}.npy")
        clean_hs = torch.from_numpy(clean_hs_np)
        _residual_stream_patching_intervention = []
        for token_idx in range(min_token_len):
            with model.trace(prompt_literal) as tracer:
                model.transformer.h[layer_idx].output[0][:, token_idx, :] = clean_hs[:, token_idx, :]
                patched_logits = model.lm_head.output
                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx])
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                _residual_stream_patching_intervention.append(patched_result.item().save())
                
        residual_stream_patching_intervention.append(_residual_stream_patching_intervention)
    clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
    clean_decoded_tokens = [model.tokenizer.decode(token) for token in clean_tokens[:min_token_len]]
    token_labels = [f"{token}_{i}" for i, token in enumerate(clean_decoded_tokens)]
    
    fig = plot_ioi_patching_results(model, residual_stream_patching_intervention, token_labels, prompt_idiom, "Patching GPT-2-small Residual Stream on Idiomatic Prompts")
    return fig


def run_residual_stream_patching(dataset, N_LAYERS):
    for pair in dataset:
        prompt_idiom = pair["prompt_idiom"]
        prompt_literal = pair["prompt_literal"]
        correct_answer = pair["correct_answer"]
        incorrect_answer = pair["incorrect_answer"]

        # tokens
        idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
        literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
        min_token_len = min(len(idiom_tokens), len(literal_tokens))

        correct_token = model.tokenizer.encode(correct_answer)[0]
        incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

        correct_answer_idx = correct_token
        incorrect_answer_idx = incorrect_token
        residual_stream_patching(N_LAYERS, prompt_idiom, prompt_literal, correct_answer_idx, incorrect_answer_idx, min_token_len)


def average_residual_stream_patching(N_LAYERS, dataset):
    """ Replaces the residual stream at the end of each layer with the residual stream from the clean prompt and plot the averaged results"""
    
    
    BEFORE = 8
    AFTER = 4
    WINDOW_SIZE = BEFORE + AFTER + 1

    total_results = np.zeros((N_LAYERS, WINDOW_SIZE))
    counts = np.zeros((N_LAYERS, WINDOW_SIZE))

    # total_kl = np.zeros((N_LAYERS, WINDOW_SIZE))
    # kl_counts = np.zeros((N_LAYERS, WINDOW_SIZE))
    
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            print(len(idiom_tokens), len(literal_tokens))
            min_token_len = min(len(idiom_tokens), len(literal_tokens))
            print(min_token_len)

            and_token = model.tokenizer.encode(",")[0]

            # Find position of "and"
            try:
                and_token_idx = (literal_tokens == and_token).nonzero(as_tuple=True)[0].item()
            except (RuntimeError, ValueError, IndexError):
                print(f"Skipping {idiom_id}: 'and' not found.")
                continue
            
            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                # clean run
                with model.trace(prompt_idiom) as tracer:
                    clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                    clean_hs = [model.transformer.h[i].output[0].save() for i in range(N_LAYERS)]
                    clean_logits = model.lm_head.output.save()
                    clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                
                clean_logits_value = clean_logits[0, :min_token_len, :].detach().cpu().numpy()
                np.save(f"data/clean_logits_value.npy", clean_logits_value)
                # Convert Proxy objects to numpy arrays after trace context exits
                clean_hs_np = [act.value.detach().cpu().numpy() for act in clean_hs]
                for layer_idx, act in enumerate(clean_hs_np):
                    np.save(f"data/clean_hidden_state_{layer_idx}.npy", act)
                
                # corrupted run
                with model.trace(prompt_literal) as tracer:
                    corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                    corrupted_logits = model.lm_head.output
                    corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()

                # patching within the window
                
                for layer_idx in range(N_LAYERS):
                    clean_hs_np = np.load(f"data/clean_hidden_state_{layer_idx}.npy")
                    clean_hs = torch.from_numpy(clean_hs_np)
                    
                    for offset in range(-BEFORE, AFTER + 1):
                        token_idx = and_token_idx + offset
                        matrix_col = offset + BEFORE
                        if 0<= token_idx < min_token_len:

                            with model.trace(prompt_literal) as tracer:
                                model.transformer.h[layer_idx].output[0][:, token_idx, :] = clean_hs[:, token_idx, :]
                                patched_logits = model.lm_head.output.save()
                                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx])
                                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                                patched_result_saved = patched_result.save()
                                
                            #     # Align sequence lengths by taking only the overlapping positions
                            # clean_logits_value = np.load(f"data/clean_logits_value.npy")
                            # clean_logits_value = torch.from_numpy(clean_logits_value)
                            # patched_logits_value = patched_logits[0, :min_token_len, :].detach().cpu().numpy()
                            # patched_logits_value = torch.from_numpy(patched_logits_value)
                            # kl_value = mean_sequence_kl(
                            #     clean_logits_value,
                            #     patched_logits_value
                            # ).item()


                            # total_kl[layer_idx, matrix_col] += kl_value
                            # kl_counts[layer_idx, matrix_col] += 1


                            # Extract the value after the trace context exits
                            total_results[layer_idx, matrix_col] += patched_result_saved.item()
                            counts[layer_idx, matrix_col] += 1
    
    return total_results, counts


# def mean_sequence_kl(clean_logits, patched_logits):
#     """
#     Computes mean KL(clean || patched) over the full output sequence.
#     """
#     clean_log_probs = F.log_softmax(clean_logits, dim=-1)
#     patched_log_probs = F.log_softmax(patched_logits, dim=-1)

#     kl_per_token = F.kl_div(
#         patched_log_probs,
#         clean_log_probs,
#         log_target=True,
#         reduction="none"
#     ).sum(dim=-1)  # (seq_len,)

#     return kl_per_token.mean()




def run_average_residual_stream_patching(dataset, N_LAYERS):
    
        total_results, counts = average_residual_stream_patching(N_LAYERS, dataset)
        avg_results = np.divide(total_results, counts, out=np.zeros_like(total_results), where=counts!=0)
        # Use first idiom's first pair for labels
        prompt_idiom = dataset[0]["pairs"][0]["prompt_idiom"]
        
        idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
        and_token = model.tokenizer.encode(",")[0]
        and_token_idx = (idiom_tokens == and_token).nonzero(as_tuple=True)[0].item()

        labels = [model.tokenizer.decode(token) for token in idiom_tokens[and_token_idx-8:and_token_idx+5]]
        # Calculate the absolute maximum to make the scale symmetric
        y_labels = list(range(1, N_LAYERS + 1))
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            avg_results,
            xticklabels=labels,
            yticklabels=y_labels,
            cmap="RdBu",
            cbar_kws={'label': 'Norm. Logit Diff'},
            center=0
        )
        plt.xlabel("Token position")
        plt.ylabel("Layer")
        plt.title("Dataset-Averaged Activation Patching")
        plt.savefig("figures/averaged_residual_stream_patching_results_multiple.png")
        plt.savefig("figures/averaged_residual_stream_patching_results_multiple.eps")
        plt.gca() # Often helpful to have Layer 0 at the bottom
        plt.show()
        return plt.gcf()



def attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, prompt_idiom, prompt_literal, correct_answer_idx, incorrect_answer_idx, min_token_len):
    """ Replaces the attention head at the end of each layer with the attention head from the clean prompt """
    batch = 1
    # clean run
    with model.trace() as tracer:
        with tracer.invoke(prompt_idiom) as invoker:
            clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            z_hs = {}
            for layer_idx in range(N_LAYERS):
                z = model.transformer.h[layer_idx].attn.c_proj.input
                z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                for head_idx in range(N_HEADS):
                    z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
            
            clean_logits = model.lm_head.output
            clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
            print(f"clean logit diff {clean_logit_diff}")
    
    # Extract values from Proxy objects after trace context exits
    z_hs_np = {}
    for (layer_idx, head_idx), proxy_obj in z_hs.items():
        z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
    
    # Now pickle the actual numpy arrays
    z_hs_file = open("z_hs", "wb")
    pickle.dump(z_hs_np, z_hs_file)
    z_hs_file.close()

    # corrupted run
    with model.trace() as tracer:
        with tracer.invoke(prompt_literal) as invoker:
            corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            corrupted_logits = model.lm_head.output
            corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
            print(f"corrupt logit diff {corrupted_logit_diff}")
    
    
    #patching
    attention_head_patching_intervention = []
    #load pickle
    z_hs_file = open("z_hs", "rb")
    z_hs_np = pickle.load(z_hs_file)
    z_hs_file.close()
    
    # Convert numpy arrays back to torch tensors
    z_hs = {}
    for (layer_idx, head_idx), np_array in z_hs_np.items():
        z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
    
    with model.trace() as tracer:
        for layer_idx in range(N_LAYERS):
            _attention_head_patching_intervention = []
            for head_idx in range(N_HEADS):
                with tracer.invoke(prompt_literal) as invoker:
                    z = model.transformer.h[layer_idx].attn.c_proj.input
                    z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                    # Only patch the overlapping sequence length (min_token_len)
                    z_corrupt[:, :min_token_len, head_idx, :] = z_hs[layer_idx, head_idx]
                    patched_logits = model.lm_head.output
                    patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
                    patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                    _attention_head_patching_intervention.append(patched_result.item().save())
            attention_head_patching_intervention.append(_attention_head_patching_intervention)
    clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
    clean_decoded_tokens = [model.tokenizer.decode(token) for token in clean_tokens[:min_token_len]]
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    fig = plot_ioi_patching_results_attention(model, attention_head_patching_intervention, x_labels, prompt_idiom, "Patching GPT-2-small Attention Head on Idiomatic Prompts")
    return fig
    
def run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS):
    for pair in dataset:
        prompt_idiom = pair["prompt_idiom"]
        prompt_literal = pair["prompt_literal"]
        correct_answer = pair["correct_answer"]
        incorrect_answer = pair["incorrect_answer"]

        # tokens
        idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
        literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
        min_token_len = min(len(idiom_tokens), len(literal_tokens))

        correct_token = model.tokenizer.encode(correct_answer)[0]
        incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

        correct_answer_idx = correct_token
        incorrect_answer_idx = incorrect_token

        attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, prompt_idiom, prompt_literal, correct_answer_idx, incorrect_answer_idx, min_token_len)


def average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset):
    accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
    # total_kl = np.zeros((N_LAYERS, N_HEADS))
    # kl_counts = np.zeros((N_LAYERS, N_HEADS))
    
    # Count total number of answer combinations (idioms * pairs * 3 answers)
    total_combinations = 0
    
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            min_token_len = min(len(idiom_tokens), len(literal_tokens))

            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                total_combinations += 1

                # clean run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom) as invoker:
                        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                        z_hs = {}
                        for layer_idx in range(N_LAYERS):
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                            for head_idx in range(N_HEADS):
                                z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
                        clean_logits = model.lm_head.output.save()
                        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"clean logit diff {clean_logit_diff}")
                clean_logits_value = clean_logits[0, :min_token_len, :].detach().cpu().numpy()
                np.save(f"data/clean_logits_value.npy", clean_logits_value)
                # Extract values from Proxy objects after trace context exits
                z_hs_np = {}
                for (layer_idx, head_idx), proxy_obj in z_hs.items():
                    z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
                # Now pickle the actual numpy arrays
                z_hs_file = open("z_hs", "wb")
                pickle.dump(z_hs_np, z_hs_file)
                z_hs_file.close()

                # corrupted run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_literal) as invoker:
                        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
                #patching
                attention_head_patching_intervention = []
                #load pickle
                z_hs_file = open("z_hs", "rb")
                z_hs_np = pickle.load(z_hs_file)
                z_hs_file.close()
                
                # Convert numpy arrays back to torch tensors
                z_hs = {}
                for (layer_idx, head_idx), np_array in z_hs_np.items():
                    z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
                # Compute actual_seq_len outside trace context to avoid Proxy issues
                # Get saved sequence length from z_hs (same for all heads in a layer)
                sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
                if len(sample_z_hs.shape) == 2:
                    saved_seq_len = sample_z_hs.shape[0]
                else:
                    saved_seq_len = sample_z_hs.shape[1]
                # Use min_token_len as the current sequence length (it's already the min of idiom and literal)
                actual_seq_len = min(saved_seq_len, min_token_len)
                
                # Prepare z_hs tensors with proper shapes
                z_hs_prepared = {}
                for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
                    if len(z_hs_tensor.shape) == 2:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
                    else:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
                patched_logits_dict = {}
                with model.trace() as tracer:
                    for layer_idx in range(N_LAYERS):
                        _attention_head_patching_intervention = []
                        for head_idx in range(N_HEADS):
                            with tracer.invoke(prompt_literal) as invoker:
                                z = model.transformer.h[layer_idx].attn.c_proj.input
                                z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                                # Only patch the overlapping sequence length
                                z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]
                                patched_logits = model.lm_head.output.save()
                                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
                                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                                patched_result_saved = patched_result.save()
                                # Save the logits slice we need for KL computation
                                patched_logits_dict[(layer_idx, head_idx)] = patched_logits[0, :min_token_len, :].save()
                            accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
                
                # # Compute KL divergence after trace context exits
                # clean_logits_value = np.load(f"data/clean_logits_value.npy")
                # clean_logits_value = torch.from_numpy(clean_logits_value)
                # for layer_idx in range(N_LAYERS):
                #     for head_idx in range(N_HEADS):
                #         patched_logits_value = patched_logits_dict[(layer_idx, head_idx)].value.detach().cpu().numpy()
                #         patched_logits_value = torch.from_numpy(patched_logits_value)
                #         kl_value = mean_sequence_kl(
                #             clean_logits_value,
                #             patched_logits_value
                #         ).item()
                #         total_kl[layer_idx, head_idx] += kl_value
                #         kl_counts[layer_idx, head_idx] += 1
        
    average_patching_results = accumulated_results / total_combinations
    flat_results = average_patching_results.flatten()
    top_values, top_indices = torch.topk(flat_results, 36)

    # Convert flat indices back to (layer, head) tuples
    top_heads = []
    for idx in top_indices:
        layer = idx.item() // N_HEADS
        head = idx.item() % N_HEADS
        top_heads.append((layer, head))

    print(f"Top 36 heads to patch/zero: {top_heads}")
    # save as a json file
    with open(f"top_36_heads_gpt2.json", "w") as f:
        json.dump(top_heads, f)
    return average_patching_results




def run_average_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS):
    average_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = "averaged_attention_multiple"
    marginalised_layers = torch.mean(average_patching_results, dim=1)
    # plot the marginalised layers
    y_labels = list(range(1, N_LAYERS + 1))
    plt.figure(figsize=(10, 6))
    plt.bar(y_labels, marginalised_layers.tolist(), color='skyblue', edgecolor='navy')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Normalized Logit Difference')
    plt.title(f'Marginalised Layer Importance (Average of {len(dataset)} Idioms for GPT2)')
    plt.xticks(y_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"figures/marginalised_layers_gpt2_multiple.png")
    plt.savefig(f"figures/marginalised_layers_gpt2_multiple.eps")
    plt.show()
    # calculate the highest value per layer
    highest_values = torch.max(average_patching_results, dim=1)
    highest_values_list = highest_values.values
    # plot the highest values per layer
    plt.figure(figsize=(10, 6))
    plt.bar(y_labels, highest_values_list.tolist(), color='skyblue', edgecolor='navy')
    plt.xlabel('Layer Index')
    plt.ylabel('Highest Normalized Logit Difference')
    plt.title(f'Highest Importance per Layer (Average of {len(dataset)} Idioms for GPT2)')
    plt.xticks(y_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"figures/highest_values_gpt2_multiple.png")
    plt.savefig(f"figures/highest_values_gpt2_multiple.eps")
    fig = plot_ioi_patching_results_attention(model, average_patching_results.tolist(), x_labels, prompt_idiom, f"Average Attention Head Patching across {len(dataset)} Idioms")
    return fig



# patching at the MLP layers of GPT

def average_mlp_patching(N_LAYERS, dataset):
    """ patch the MLP layers"""

    
    
    BEFORE = 8
    AFTER = 4
    WINDOW_SIZE = BEFORE + AFTER + 1

    total_results = np.zeros((N_LAYERS, WINDOW_SIZE))
    counts = np.zeros((N_LAYERS, WINDOW_SIZE))
    mlp_dim_totals = None
    mlp_dim_counts = None

    # total_kl = np.zeros((N_LAYERS, WINDOW_SIZE))
    # kl_counts = np.zeros((N_LAYERS, WINDOW_SIZE))
    
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            print(len(idiom_tokens), len(literal_tokens))
            min_token_len = min(len(idiom_tokens), len(literal_tokens))
            print(min_token_len)

            and_token = model.tokenizer.encode(",")[0]

            # Find position of "and"
            try:
                and_token_idx = (literal_tokens == and_token).nonzero(as_tuple=True)[0].item()
            except (RuntimeError, ValueError, IndexError):
                print(f"Skipping {idiom_id}: 'and' not found.")
                continue
            
            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                # clean run
                with model.trace(prompt_idiom) as tracer:
                    clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                    clean_mlp = [model.transformer.h[i].mlp.c_proj.input.save() for i in range(N_LAYERS)]
                    clean_logits = model.lm_head.output.save()
                    clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                
                clean_logits_value = clean_logits[0, :min_token_len, :].detach().cpu().numpy()
                np.save(f"data/clean_logits_value.npy", clean_logits_value)
                # Convert Proxy objects to numpy arrays after trace context exits
                clean_mlp_np = [act.value.detach().cpu().numpy() for act in clean_mlp]
                for layer_idx, act in enumerate(clean_mlp_np):
                    np.save(f"data/clean_mlp_{layer_idx}.npy", act)
                
                # corrupted run
                with model.trace(prompt_literal) as tracer:
                    corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                    corrupted_mlp = [model.transformer.h[i].mlp.c_proj.input.save() for i in range(N_LAYERS)]
                    corrupted_logits = model.lm_head.output
                    corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()

                if answer_idx == 0:
                    corrupted_mlp_np = [act.value.detach().cpu().numpy() for act in corrupted_mlp]
                    if mlp_dim_totals is None:
                        mlp_dim = clean_mlp_np[0].shape[-1]
                        mlp_dim_totals = np.zeros((N_LAYERS, mlp_dim))
                        mlp_dim_counts = np.zeros((N_LAYERS, mlp_dim))

                    if 0 <= and_token_idx < min_token_len:
                        for layer_idx in range(N_LAYERS):
                            clean_vec = clean_mlp_np[layer_idx][0, and_token_idx, :]
                            corrupted_vec = corrupted_mlp_np[layer_idx][0, and_token_idx, :]
                            mlp_dim_totals[layer_idx, :] += np.abs(clean_vec - corrupted_vec)
                            mlp_dim_counts[layer_idx, :] += 1

                # patching within the window
                
                for layer_idx in range(N_LAYERS):
                    clean_mlp_np = np.load(f"data/clean_mlp_{layer_idx}.npy")
                    clean_mlp = torch.from_numpy(clean_mlp_np)
                    
                    for offset in range(-BEFORE, AFTER + 1):
                        token_idx = and_token_idx + offset
                        matrix_col = offset + BEFORE
                        if 0<= token_idx < min_token_len:

                            with model.trace(prompt_literal) as tracer:
                                model.transformer.h[layer_idx].mlp.c_proj.input[:, token_idx, :] = clean_mlp[:, token_idx, :]
                                patched_logits = model.lm_head.output.save()
                                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx])
                                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                                patched_result_saved = patched_result.save()
                                
                        

                            # Extract the value after the trace context exits
                            total_results[layer_idx, matrix_col] += patched_result_saved.item()
                            counts[layer_idx, matrix_col] += 1
    
    return total_results, counts, mlp_dim_totals, mlp_dim_counts

def run_average_mlp_patching(dataset, N_LAYERS):
    total_results, counts, mlp_dim_totals, mlp_dim_counts = average_mlp_patching(N_LAYERS, dataset)
    avg_results = np.divide(total_results, counts, out=np.zeros_like(total_results), where=counts!=0)
    avg_mlp_dim_scores = np.divide(
        mlp_dim_totals,
        mlp_dim_counts,
        out=np.zeros_like(mlp_dim_totals),
        where=mlp_dim_counts != 0
    )

    # get the top mlp components to patch
    BEFORE = 8
    AFTER = 4
    WINDOW_SIZE = BEFORE + AFTER + 1
    #flat_results = avg_results.flatten()
    flat_results = torch.from_numpy(avg_results.flatten()).float()

    top_values, top_indices = torch.topk(flat_results, 14)
    top_components = []
    for idx in top_indices:
        layer = idx.item() // WINDOW_SIZE
        component = idx.item() % WINDOW_SIZE
        top_components.append((layer, component))
    print(f"Top 14 components to patch: {top_components}")
    # save
    with open(f"top_14_mlp_components_gpt2.json", "w") as f:
        json.dump(top_components, f)

    top_values, top_indices = torch.topk(flat_results, 36)
    top_components = []
    for idx in top_indices:
        layer = idx.item() // WINDOW_SIZE
        component = idx.item() % WINDOW_SIZE
        top_components.append((layer, component))
    print(f"Top 36 components to patch: {top_components}")
    # save
    with open(f"top_36_mlp_components_gpt2.json", "w") as f:
        json.dump(top_components, f)

    # Top actual MLP neuron dimensions: [layer, mlp_component]
    flat_dim_scores = torch.from_numpy(avg_mlp_dim_scores.flatten()).float()
    mlp_dim_size = avg_mlp_dim_scores.shape[1]

    top_values, top_indices = torch.topk(flat_dim_scores, 14)
    top_mlp_dimensions = []
    for idx in top_indices:
        layer = idx.item() // mlp_dim_size
        mlp_component = idx.item() % mlp_dim_size
        top_mlp_dimensions.append((layer, mlp_component))
    print(f"Top 14 MLP neuron dimensions: {top_mlp_dimensions}")
    with open("top_14_mlp_dimensions_gpt2.json", "w") as f:
        json.dump(top_mlp_dimensions, f)

    top_values, top_indices = torch.topk(flat_dim_scores, 36)
    top_mlp_dimensions = []
    for idx in top_indices:
        layer = idx.item() // mlp_dim_size
        mlp_component = idx.item() % mlp_dim_size
        top_mlp_dimensions.append((layer, mlp_component))
    print(f"Top 36 MLP neuron dimensions: {top_mlp_dimensions}")
    with open("top_36_mlp_dimensions_gpt2.json", "w") as f:
        json.dump(top_mlp_dimensions, f)


    # Use first idiom's first pair for labels
    prompt_idiom = dataset[0]["pairs"][0]["prompt_idiom"]
        
    idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
    and_token = model.tokenizer.encode(",")[0]
    and_token_idx = (idiom_tokens == and_token).nonzero(as_tuple=True)[0].item()

    labels = [model.tokenizer.decode(token) for token in idiom_tokens[and_token_idx-8:and_token_idx+5]]
    # Calculate the absolute maximum to make the scale symmetric
    y_labels = list(range(1, N_LAYERS + 1))
    plt.figure(figsize=(12, 8))
    sns.heatmap(
            avg_results,
            xticklabels=labels,
            yticklabels=y_labels,
            cmap="RdBu",
            cbar_kws={'label': 'Norm. Logit Diff'},
            center=0
        )
    plt.xlabel("Token position")
    plt.ylabel("Layer")
    plt.title("Dataset-Averaged MLP Activation Patching")
    plt.savefig("figures/averaged_mlp_patching_results_multiple.png")
    plt.savefig("figures/averaged_mlp_patching_results_multiple.eps")
    plt.gca() # Often helpful to have Layer 0 at the bottom
    plt.show()
    return plt.gcf()




def clean_run_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)

    flat_results = averaged_patching_results.flatten()
    top_values, top_indices = torch.topk(flat_results, k)

    # Convert flat indices back to (layer, head) tuples

    # Convert flat indices back to (layer, head) tuples
    top_heads = []
    for idx in top_indices:
        layer = idx.item() // N_HEADS
        head = idx.item() % N_HEADS
        top_heads.append((layer, head))

    print(f"Top {k} heads to patch/zero: {top_heads}")

    accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
    # total_kl = torch.zeros((N_LAYERS, N_HEADS))
    # kl_counts = torch.zeros((N_LAYERS, N_HEADS))
    
    total_combinations = 0
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            min_token_len = min(len(idiom_tokens), len(literal_tokens))

            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                total_combinations += 1

                # clean run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom) as invoker:
                        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                        z_hs = {}
                        for layer_idx in range(N_LAYERS):
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                            for head_idx in range(N_HEADS):
                                z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
                        clean_logits = model.lm_head.output.save()
                        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"clean logit diff {clean_logit_diff}")
                        # Save the clean logits slice we need for KL computation
                        clean_logits_slice = clean_logits[0, :min_token_len, :].save()
                # Extract values from Proxy objects after trace context exits
                z_hs_np = {}
                for (layer_idx, head_idx), proxy_obj in z_hs.items():
                    z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                # Zero top_heads in the clean-run activations before saving (ablation)
                for (layer_idx, head_idx) in top_heads:
                    z_hs_np[layer_idx, head_idx] = np.zeros_like(z_hs_np[layer_idx, head_idx])
                
                # Now pickle the actual numpy arrays
                z_hs_file = open("z_hs_top_k", "wb")
                pickle.dump(z_hs_np, z_hs_file)
                z_hs_file.close()

                # corrupted run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_literal) as invoker:
                        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
                #patching
                
                #load pickle
                z_hs_file = open("z_hs_top_k", "rb")
                z_hs_np = pickle.load(z_hs_file)
                z_hs_file.close()
                
                # Convert numpy arrays back to torch tensors
                z_hs = {}
                for (layer_idx, head_idx), np_array in z_hs_np.items():
                    z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
                # Compute actual_seq_len outside trace context to avoid Proxy issues
                sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
                if len(sample_z_hs.shape) == 2:
                    saved_seq_len = sample_z_hs.shape[0]
                else:
                    saved_seq_len = sample_z_hs.shape[1]
                actual_seq_len = min(saved_seq_len, min_token_len)
                
                # Prepare z_hs tensors with proper shapes
                z_hs_prepared = {}
                for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
                    if len(z_hs_tensor.shape) == 2:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
                    else:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
                patched_logits_dict = {}
                with model.trace() as tracer:
                    for layer_idx in range(N_LAYERS):
                        for head_idx in range(N_HEADS):
                            with tracer.invoke(prompt_literal) as invoker:
                                z = model.transformer.h[layer_idx].attn.c_proj.input
                                z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                                # Only patch the overlapping sequence length (z_hs_prepared already has top_heads zeroed)
                                z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]

                                patched_logits = model.lm_head.output.save()
                                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
                                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                                patched_result_saved = patched_result.save()
                                patched_logits_dict[(layer_idx, head_idx)] = patched_logits[0, :min_token_len, :].save()
                            accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
                
                # # Compute KL divergence after trace context exits
                # # Use the clean logits slice saved for this specific pair (with matching min_token_len)
                # clean_logits_value = clean_logits_slice.detach().cpu().numpy()
                # clean_logits_value = torch.from_numpy(clean_logits_value)
                # for layer_idx in range(N_LAYERS):
                #     for head_idx in range(N_HEADS):
                #         patched_logits_value = patched_logits_dict[(layer_idx, head_idx)].value.detach().cpu().numpy()
                #         patched_logits_value = torch.from_numpy(patched_logits_value)
                #         kl_value = mean_sequence_kl(
                #             clean_logits_value,
                #             patched_logits_value
                #         ).item()
                #         total_kl[layer_idx, head_idx] += kl_value
                #         kl_counts[layer_idx, head_idx] += 1
    average_patching_results = accumulated_results / total_combinations
    return average_patching_results

def run_clean_run_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    average_patching_results = clean_run_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k)
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = f"top_{k}_k_clean_run_attention_head_ablation_multiple"
    fig = plot_ioi_patching_results_attention(model, average_patching_results.tolist(), x_labels, prompt_idiom, f"Top {k} Attention Head Ablation across {len(dataset)} Idioms")
    return fig


def clean_run_random_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)

    flat_results = averaged_patching_results.flatten()
    # top_values, top_indices = torch.topk(flat_results, k)
    top_indices = torch.randperm(len(flat_results))[:k]

    # Convert flat indices back to (layer, head) tuples
    top_heads = []
    for idx in top_indices:
        layer = idx.item() // N_HEADS
        head = idx.item() % N_HEADS
        top_heads.append((layer, head))

    print(f"Random {k} heads to patch/zero: {top_heads}")

    accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
    # total_kl = torch.zeros((N_LAYERS, N_HEADS))
    # kl_counts = torch.zeros((N_LAYERS, N_HEADS))
    
    total_combinations = 0
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            min_token_len = min(len(idiom_tokens), len(literal_tokens))

            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                total_combinations += 1

                # clean run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom) as invoker:
                        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                        z_hs = {}
                        for layer_idx in range(N_LAYERS):
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                            for head_idx in range(N_HEADS):
                                z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
                        clean_logits = model.lm_head.output.save()
                        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"clean logit diff {clean_logit_diff}")
                        # Save the clean logits slice we need for KL computation
                        clean_logits_slice = clean_logits[0, :min_token_len, :].save()
                # Extract values from Proxy objects after trace context exits
                z_hs_np = {}
                for (layer_idx, head_idx), proxy_obj in z_hs.items():
                    z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                # Zero top_heads in the clean-run activations before saving (ablation)
                for (layer_idx, head_idx) in top_heads:
                    z_hs_np[layer_idx, head_idx] = np.zeros_like(z_hs_np[layer_idx, head_idx])
                
                # Now pickle the actual numpy arrays
                z_hs_file = open("z_hs_top_k", "wb")
                pickle.dump(z_hs_np, z_hs_file)
                z_hs_file.close()

                # corrupted run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_literal) as invoker:
                        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
                #patching
                
                #load pickle
                z_hs_file = open("z_hs_top_k", "rb")
                z_hs_np = pickle.load(z_hs_file)
                z_hs_file.close()
                
                # Convert numpy arrays back to torch tensors
                z_hs = {}
                for (layer_idx, head_idx), np_array in z_hs_np.items():
                    z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
                # Compute actual_seq_len outside trace context to avoid Proxy issues
                sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
                if len(sample_z_hs.shape) == 2:
                    saved_seq_len = sample_z_hs.shape[0]
                else:
                    saved_seq_len = sample_z_hs.shape[1]
                actual_seq_len = min(saved_seq_len, min_token_len)
                
                # Prepare z_hs tensors with proper shapes
                z_hs_prepared = {}
                for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
                    if len(z_hs_tensor.shape) == 2:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
                    else:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
                patched_logits_dict = {}
                with model.trace() as tracer:
                    for layer_idx in range(N_LAYERS):
                        for head_idx in range(N_HEADS):
                            with tracer.invoke(prompt_literal) as invoker:
                                z = model.transformer.h[layer_idx].attn.c_proj.input
                                z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                                # Only patch the overlapping sequence length (z_hs_prepared already has top_heads zeroed)
                                z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]

                                patched_logits = model.lm_head.output.save()
                                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
                                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                                patched_result_saved = patched_result.save()
                                patched_logits_dict[(layer_idx, head_idx)] = patched_logits[0, :min_token_len, :].save()
                            accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
                
                # # Compute KL divergence after trace context exits
                # # Use the clean logits slice saved for this specific pair (with matching min_token_len)
                # clean_logits_value = clean_logits_slice.detach().cpu().numpy()
                # clean_logits_value = torch.from_numpy(clean_logits_value)
                # for layer_idx in range(N_LAYERS):
                #     for head_idx in range(N_HEADS):
                #         patched_logits_value = patched_logits_dict[(layer_idx, head_idx)].value.detach().cpu().numpy()
                #         patched_logits_value = torch.from_numpy(patched_logits_value)
                #         kl_value = mean_sequence_kl(
                #             clean_logits_value,
                #             patched_logits_value
                #         ).item()
                #         total_kl[layer_idx, head_idx] += kl_value
                #         kl_counts[layer_idx, head_idx] += 1
    average_patching_results = accumulated_results / total_combinations
    return average_patching_results

def run_clean_run_random_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    average_patching_results = clean_run_random_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k)
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = f"top_{k}_k_clean_run_random_attention_head_ablation_multiple"
    fig = plot_ioi_patching_results_attention(model, average_patching_results.tolist(), x_labels, prompt_idiom, f"Top {k} Attention Head Ablation across {len(dataset)} Idioms")
    return fig

def top_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)
     
    flat_results = averaged_patching_results.flatten()
    top_values, top_indices = torch.topk(flat_results, k)

    # Convert flat indices back to (layer, head) tuples

    # Convert flat indices back to (layer, head) tuples
    top_heads = []
    for idx in top_indices:
        layer = idx.item() // N_HEADS
        head = idx.item() % N_HEADS
        top_heads.append((layer, head))

    print(f"Top {k} heads to patch/zero: {top_heads}")

    accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
    # total_kl = torch.zeros((N_LAYERS, N_HEADS))
    # kl_counts = torch.zeros((N_LAYERS, N_HEADS))
    
    total_combinations = 0
    
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            min_token_len = min(len(idiom_tokens), len(literal_tokens))

            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                total_combinations += 1

                # clean run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom) as invoker:
                        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                        z_hs = {}
                        for layer_idx in range(N_LAYERS):
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                            for head_idx in range(N_HEADS):
                                z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
                        clean_logits = model.lm_head.output.save()
                        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"clean logit diff {clean_logit_diff}")
                        # Save the clean logits slice we need for KL computation
                        clean_logits_slice = clean_logits[0, :min_token_len, :].save()
                # Extract values from Proxy objects after trace context exits
                z_hs_np = {}
                for (layer_idx, head_idx), proxy_obj in z_hs.items():
                    z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
                # Now pickle the actual numpy arrays
                z_hs_file = open("z_hs_top_k", "wb")
                pickle.dump(z_hs_np, z_hs_file)
                z_hs_file.close()

                # corrupted run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_literal) as invoker:
                        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
                #patching
                
                #load pickle
                z_hs_file = open("z_hs_top_k", "rb")
                z_hs_np = pickle.load(z_hs_file)
                z_hs_file.close()
                
                # Convert numpy arrays back to torch tensors
                z_hs = {}
                for (layer_idx, head_idx), np_array in z_hs_np.items():
                    z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
                # Compute actual_seq_len outside trace context to avoid Proxy issues
                sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
                if len(sample_z_hs.shape) == 2:
                    saved_seq_len = sample_z_hs.shape[0]
                else:
                    saved_seq_len = sample_z_hs.shape[1]
                actual_seq_len = min(saved_seq_len, min_token_len)
                
                # Prepare z_hs tensors with proper shapes
                z_hs_prepared = {}
                for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
                    if len(z_hs_tensor.shape) == 2:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
                    else:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
                patched_logits_dict = {}
                with model.trace() as tracer:
                    for layer_idx in range(N_LAYERS):
                        for head_idx in range(N_HEADS):
                            with tracer.invoke(prompt_literal) as invoker:
                                z = model.transformer.h[layer_idx].attn.c_proj.input
                                z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                                # Only patch the overlapping sequence length
                                z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]
                                if (layer_idx, head_idx) in top_heads:
                                    z_corrupt[:, :actual_seq_len, head_idx, :] = 0


                                patched_logits = model.lm_head.output.save()
                                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
                                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                                patched_result_saved = patched_result.save()
                                patched_logits_dict[(layer_idx, head_idx)] = patched_logits[0, :min_token_len, :].save()
                            accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
                
                # # Compute KL divergence after trace context exits
                # # Use the clean logits slice saved for this specific pair (with matching min_token_len)
                # clean_logits_value = clean_logits_slice.detach().cpu().numpy()
                # clean_logits_value = torch.from_numpy(clean_logits_value)
                # for layer_idx in range(N_LAYERS):
                #     for head_idx in range(N_HEADS):
                #         patched_logits_value = patched_logits_dict[(layer_idx, head_idx)].value.detach().cpu().numpy()
                #         patched_logits_value = torch.from_numpy(patched_logits_value)
                #         kl_value = mean_sequence_kl(
                #             clean_logits_value,
                #             patched_logits_value
                #         ).item()
                #         total_kl[layer_idx, head_idx] += kl_value
                #         kl_counts[layer_idx, head_idx] += 1
    average_patching_results = accumulated_results / total_combinations
    return average_patching_results


def run_top_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    average_patching_results = top_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k)
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = f"top_{k}_k_attention_head_ablation_multiple"
    fig = plot_ioi_patching_results_attention(model, average_patching_results.tolist(), x_labels, prompt_idiom, f"Top {k} Attention Head Ablation across {len(dataset)} Idioms")
    return fig


def random_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)
     
    flat_results = averaged_patching_results.flatten()
    # top_values, top_indices = torch.topk(flat_results, k)
    top_indices = torch.randperm(len(flat_results))[:k]

    # Convert flat indices back to (layer, head) tuples
    top_heads = []
    for idx in top_indices:
        layer = idx.item() // N_HEADS
        head = idx.item() % N_HEADS
        top_heads.append((layer, head))

    print(f"Random {k} heads to patch/zero: {top_heads}")

    accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
    # total_kl = torch.zeros((N_LAYERS, N_HEADS))
    # kl_counts = torch.zeros((N_LAYERS, N_HEADS))
    
    total_combinations = 0
    
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            min_token_len = min(len(idiom_tokens), len(literal_tokens))

            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                total_combinations += 1

                # clean run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom) as invoker:
                        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                        z_hs = {}
                        for layer_idx in range(N_LAYERS):
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                            for head_idx in range(N_HEADS):
                                z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
                        clean_logits = model.lm_head.output.save()
                        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"clean logit diff {clean_logit_diff}")
                        # Save the clean logits slice we need for KL computation
                        clean_logits_slice = clean_logits[0, :min_token_len, :].save()
                # Extract values from Proxy objects after trace context exits
                z_hs_np = {}
                for (layer_idx, head_idx), proxy_obj in z_hs.items():
                    z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
                # Now pickle the actual numpy arrays
                z_hs_file = open("z_hs_top_k", "wb")
                pickle.dump(z_hs_np, z_hs_file)
                z_hs_file.close()

                # corrupted run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_literal) as invoker:
                        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
                #patching
                
                #load pickle
                z_hs_file = open("z_hs_top_k", "rb")
                z_hs_np = pickle.load(z_hs_file)
                z_hs_file.close()
                
                # Convert numpy arrays back to torch tensors
                z_hs = {}
                for (layer_idx, head_idx), np_array in z_hs_np.items():
                    z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
                # Compute actual_seq_len outside trace context to avoid Proxy issues
                sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
                if len(sample_z_hs.shape) == 2:
                    saved_seq_len = sample_z_hs.shape[0]
                else:
                    saved_seq_len = sample_z_hs.shape[1]
                actual_seq_len = min(saved_seq_len, min_token_len)
                
                # Prepare z_hs tensors with proper shapes
                z_hs_prepared = {}
                for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
                    if len(z_hs_tensor.shape) == 2:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
                    else:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
                patched_logits_dict = {}
                with model.trace() as tracer:
                    for layer_idx in range(N_LAYERS):
                        for head_idx in range(N_HEADS):
                            with tracer.invoke(prompt_literal) as invoker:
                                z = model.transformer.h[layer_idx].attn.c_proj.input
                                z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                                # Only patch the overlapping sequence length
                                z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]
                                if (layer_idx, head_idx) in top_heads:
                                    z_corrupt[:, :actual_seq_len, head_idx, :] = 0


                                patched_logits = model.lm_head.output.save()
                                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
                                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                                patched_result_saved = patched_result.save()
                                patched_logits_dict[(layer_idx, head_idx)] = patched_logits[0, :min_token_len, :].save()
                            accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
                
                # # Compute KL divergence after trace context exits
                # # Use the clean logits slice saved for this specific pair (with matching min_token_len)
                # clean_logits_value = clean_logits_slice.detach().cpu().numpy()
                # clean_logits_value = torch.from_numpy(clean_logits_value)
                # for layer_idx in range(N_LAYERS):
                #     for head_idx in range(N_HEADS):
                #         patched_logits_value = patched_logits_dict[(layer_idx, head_idx)].value.detach().cpu().numpy()
                #         patched_logits_value = torch.from_numpy(patched_logits_value)
                #         kl_value = mean_sequence_kl(
                #             clean_logits_value,
                #             patched_logits_value
                #         ).item()
                #         total_kl[layer_idx, head_idx] += kl_value
                #         kl_counts[layer_idx, head_idx] += 1
    average_patching_results = accumulated_results / total_combinations
    return average_patching_results


def run_random_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    average_patching_results = random_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k)
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = f"random_{k}_k_attention_head_ablation_multiple"
    fig = plot_ioi_patching_results_attention(model, average_patching_results.tolist(), x_labels, prompt_idiom, f"Random {k} Attention Head Ablation across {len(dataset)} Idioms")
    return fig


### need to only patch the top k heads and then print the logit difference


def patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)
     
    flat_results = averaged_patching_results.flatten()
    top_values, top_indices = torch.topk(flat_results, k)

    # Convert flat indices back to (layer, head) tuples

    # Convert flat indices back to (layer, head) tuples
    top_heads = []
    for idx in top_indices:
        layer = idx.item() // N_HEADS
        head = idx.item() % N_HEADS
        top_heads.append((layer, head))

    print(f"Top {k} heads to patch/zero: {top_heads}")

    accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
    # total_kl = torch.zeros((N_LAYERS, N_HEADS))
    # kl_counts = torch.zeros((N_LAYERS, N_HEADS))
    
    total_combinations = 0
    
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            min_token_len = min(len(idiom_tokens), len(literal_tokens))

            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                total_combinations += 1

                # clean run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom) as invoker:
                        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                        z_hs = {}
                        for layer_idx in range(N_LAYERS):
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                            for head_idx in range(N_HEADS):
                                z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
                        clean_logits = model.lm_head.output.save()
                        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"clean logit diff {clean_logit_diff}")
                clean_logits_value = clean_logits[0, :min_token_len, :].detach().cpu().numpy()
                np.save(f"data/clean_logits_value.npy", clean_logits_value)
                # Extract values from Proxy objects after trace context exits
                z_hs_np = {}
                for (layer_idx, head_idx), proxy_obj in z_hs.items():
                    z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
                # Now pickle the actual numpy arrays
                z_hs_file = open("z_hs_top_k", "wb")
                pickle.dump(z_hs_np, z_hs_file)
                z_hs_file.close()

                # corrupted run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_literal) as invoker:
                        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
                #patching
                #load pickle
                z_hs_file = open("z_hs_top_k", "rb")
                z_hs_np = pickle.load(z_hs_file)
                z_hs_file.close()
                
                # Convert numpy arrays back to torch tensors
                z_hs = {}
                for (layer_idx, head_idx), np_array in z_hs_np.items():
                    z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
                # Compute actual_seq_len outside trace context to avoid Proxy issues
                sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
                if len(sample_z_hs.shape) == 2:
                    saved_seq_len = sample_z_hs.shape[0]
                else:
                    saved_seq_len = sample_z_hs.shape[1]
                actual_seq_len = min(saved_seq_len, min_token_len)
                
                # Prepare z_hs tensors with proper shapes
                z_hs_prepared = {}
                for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
                    if len(z_hs_tensor.shape) == 2:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
                    else:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
                patched_logits_dict = {}
                # Patch each selected head independently (one forward pass per head).
                with model.trace() as tracer:
                    for (layer_idx, head_idx) in top_heads:
                        with tracer.invoke(prompt_literal) as invoker:
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_corrupt = einops.rearrange(
                                z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS
                            )
                            z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]

                            patched_logits = model.lm_head.output.save()
                            patched_logit_diff = (
                                patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]
                            ).save()
                            patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                            print(f"patched result ({layer_idx}, {head_idx}) {patched_result}")
                            patched_result_saved = patched_result.save()

                            accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
    average_patching_results = accumulated_results / total_combinations
    return average_patching_results


def run_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    average_patching_results = patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k)
    print(f"Average patching results: {average_patching_results}")
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = f"top_{k}_heads_patching_only"
    fig = plot_ioi_patching_results_attention(model, average_patching_results.tolist(), x_labels, prompt_idiom, f"Only Patching Top {k} Heads across {len(dataset)} Idioms")
    return fig

def random_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)
     
    flat_results = averaged_patching_results.flatten()
    # top_values, top_indices = torch.topk(flat_results, k)
    top_indices = torch.randperm(len(flat_results))[:k]

    # Convert flat indices back to (layer, head) tuples
    top_heads = []
    for idx in top_indices:
        layer = idx.item() // N_HEADS
        head = idx.item() % N_HEADS
        top_heads.append((layer, head))

    print(f"Random {k} heads to patch/zero: {top_heads}")

    accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
    # total_kl = torch.zeros((N_LAYERS, N_HEADS))
    # kl_counts = torch.zeros((N_LAYERS, N_HEADS))


    total_combinations = 0
    
    # Iterate through idioms
    for idiom_entry in dataset:
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        
        # For each idiom, iterate through pairs
        for pair in pairs:
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            min_token_len = min(len(idiom_tokens), len(literal_tokens))

            # For each pair, iterate through the 3 answer combinations
            for answer_idx in range(3):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                
                correct_token = model.tokenizer.encode(correct_answer)[0]
                incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                
                total_combinations += 1

                # clean run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom) as invoker:
                        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
                        z_hs = {}
                        for layer_idx in range(N_LAYERS):
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                            for head_idx in range(N_HEADS):
                                z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
                        clean_logits = model.lm_head.output.save()
                        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"clean logit diff {clean_logit_diff}")
                clean_logits_value = clean_logits[0, :min_token_len, :].detach().cpu().numpy()
                np.save(f"data/clean_logits_value.npy", clean_logits_value)
                # Extract values from Proxy objects after trace context exits
                z_hs_np = {}
                for (layer_idx, head_idx), proxy_obj in z_hs.items():
                    z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
                # Now pickle the actual numpy arrays
                z_hs_file = open("z_hs_top_k", "wb")
                pickle.dump(z_hs_np, z_hs_file)
                z_hs_file.close()

                # corrupted run
                with model.trace() as tracer:
                    with tracer.invoke(prompt_literal) as invoker:
                        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
                #patching
                #load pickle
                z_hs_file = open("z_hs_top_k", "rb")
                z_hs_np = pickle.load(z_hs_file)
                z_hs_file.close()
                
                # Convert numpy arrays back to torch tensors
                z_hs = {}
                for (layer_idx, head_idx), np_array in z_hs_np.items():
                    z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
                # Compute actual_seq_len outside trace context to avoid Proxy issues
                sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
                if len(sample_z_hs.shape) == 2:
                    saved_seq_len = sample_z_hs.shape[0]
                else:
                    saved_seq_len = sample_z_hs.shape[1]
                actual_seq_len = min(saved_seq_len, min_token_len)
                
                # Prepare z_hs tensors with proper shapes
                z_hs_prepared = {}
                for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
                    if len(z_hs_tensor.shape) == 2:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
                    else:
                        z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
                patched_logits_dict = {}
                # Patch each selected head independently (one forward pass per head).
                with model.trace() as tracer:
                    for (layer_idx, head_idx) in top_heads:
                        with tracer.invoke(prompt_literal) as invoker:
                            z = model.transformer.h[layer_idx].attn.c_proj.input
                            z_corrupt = einops.rearrange(
                                z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS
                            )
                            z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]

                            patched_logits = model.lm_head.output.save()
                            patched_logit_diff = (
                                patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]
                            ).save()
                            patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                            print(f"patched result ({layer_idx}, {head_idx}) {patched_result}")
                            patched_result_saved = patched_result.save()

                            accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
    average_patching_results = accumulated_results / total_combinations
    return average_patching_results


def run_random_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k):
    average_patching_results = random_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k)
    print(f"Average patching results: {average_patching_results}")
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = f"random_{k}_heads_patching_only"
    fig = plot_ioi_patching_results_attention(model, average_patching_results.tolist(), x_labels, prompt_idiom, f"Only Patching Random {k} Heads across {len(dataset)} Idioms")
    return fig



### MAIN

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = json.load(open(f"data/{args.dataset}"))
   
    # Print dataset structure for debugging
    for idiom_entry in dataset:
        print("--------------------------------")
        print(f"Idiom: {idiom_entry['id']}")
        for pair in idiom_entry["pairs"]:
            print(f"  Idiom prompt: {pair['prompt_idiom']}")
            print(f"  Literal prompt: {pair['prompt_literal']}")
        print("--------------------------------")

    if args.model == "gpt2":
        model = LanguageModel("openai-community/gpt2", device_map="auto")
        N_LAYERS = len(model.transformer.h)
        N_HEADS = 12
        D_MODEL = 768
        D_HEADS = D_MODEL // N_HEADS

        
        if args.intervention == "residual_stream":
            if args.averaging:
                run_average_residual_stream_patching(dataset, N_LAYERS)
            else:
                run_residual_stream_patching(dataset, N_LAYERS)
        elif args.intervention == "attention_head":
            
            if args.averaging:
                run_average_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS)
                if args.visualise_attention_head_patterns:
                    prompt_idiom = dataset[0]["pairs"][0]["prompt_idiom"]
                    visualize_attention(model, prompt_idiom, layer = 3, head = 7)
                    visualize_attention(model, prompt_idiom, layer = 8, head = 9)
                    visualize_attention(model, prompt_idiom, layer = 9, head = 5)
                if args.ablation:
                    if args.random_ablation:
                        run_random_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, args.top_k)
                    else:
                        run_top_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, args.top_k)
                if args.clean_run_ablation:
                    if args.random_ablation:
                        run_random_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, args.top_k)
                    else:
                        run_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, args.top_k)
            else:
                run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS)
        elif args.intervention == "mlp":
            if args.averaging:
                run_average_mlp_patching(dataset, N_LAYERS)
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")

    elif args.model == "llama3b":
        model = LanguageModel("meta-llama/Llama-3.2-3B", device_map="auto")
        N_LAYERS = len(model.model.layers)
        N_HEADS = 24
        D_MODEL = 3072
        D_HEADS = D_MODEL // N_HEADS
        if args.intervention == "residual_stream":
            run_residual_stream_patching(dataset, N_LAYERS)
        elif args.intervention == "attention_head":
            run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS)
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")
    else:
        raise ValueError(f"Invalid model: {args.model}")
