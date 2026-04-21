
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


def mean_sequence_kl(clean_logits, patched_logits):
    """
    Mean KL(clean || patched) over sequence positions.
    clean_logits, patched_logits: (seq_len, vocab) float tensor.
    """
    clean_logits = clean_logits.float()
    patched_logits = patched_logits.float()
    clean_log_probs = F.log_softmax(clean_logits, dim=-1)
    patched_log_probs = F.log_softmax(patched_logits, dim=-1)
    kl_per_token = F.kl_div(
        patched_log_probs,
        clean_log_probs,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)
    neg_mask = kl_per_token < 0
    if neg_mask.any():
        n = int(neg_mask.sum().item())
        bad = kl_per_token[neg_mask]
        sample = bad[:10].tolist() + (["..."] if n > 10 else [])
        print(
            f"[mean_sequence_kl] negative per-token KL: count={n}, "
            f"min={bad.min().item():.6e}, sample={sample}"
        )
    out = kl_per_token.mean()
    if out.item() < 0:
        print(f"[mean_sequence_kl] negative mean KL: {out.item():.6e}")
    else:
        print(f"[mean_sequence_kl] positive mean KL: {out.item():.6e}")
    return out


def _saved_tensor_value(saved):
    """Resolve nnsight .save() output to a torch.Tensor."""
    return saved.value if hasattr(saved, "value") else saved


def plot_patching_kl_heatmap(
    kl_matrix,
    xticklabels,
    yticklabels,
    title,
    png_path,
    eps_path,
    xlabel="Column",
    ylabel="Layer",
):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        kl_matrix,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="rocket_r",
        cbar_kws={"label": "KL(clean || patched)"},
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.savefig(eps_path)
    plt.show()
    return plt.gcf()



def plot_kl_vs_layer_line(kl_matrix, title, png_path, eps_path):
    """
    Line plot with layer on the x-axis. kl_matrix is (n_layers, n_cols); values are
    averaged across columns (token positions or heads) to get one KL value per layer.
    """
    kl_matrix = np.asarray(kl_matrix, dtype=float)
    if kl_matrix.ndim == 2:
        kl_per_layer = np.nanmean(kl_matrix, axis=1)
    else:
        kl_per_layer = np.asarray(kl_matrix).reshape(-1)
    n_layers = len(kl_per_layer)
    x = np.arange(1, n_layers + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, kl_per_layer, marker="o", color="steelblue")
    plt.xlabel("Layer")
    plt.ylabel("Mean KL(clean || patched)")
    plt.title(title)
    plt.xticks(x)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.savefig(eps_path)
    plt.show()
    return plt.gcf()


def average_residual_stream_patching(N_LAYERS, dataset):
    """ Replaces the residual stream at the end of each layer with the residual stream from the clean prompt and plot the averaged results"""
    
    
    BEFORE = 8
    AFTER = 4
    WINDOW_SIZE = BEFORE + AFTER + 1

    total_results = np.zeros((N_LAYERS, WINDOW_SIZE))
    counts = np.zeros((N_LAYERS, WINDOW_SIZE))

    total_kl = np.zeros((N_LAYERS, WINDOW_SIZE))
    kl_counts = np.zeros((N_LAYERS, WINDOW_SIZE))
    
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
                
                clean_logits_value = _saved_tensor_value(clean_logits)[0, :min_token_len, :].float()
                np.save(f"data/clean_logits_value.npy", clean_logits_value.detach().cpu().numpy())
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

                            patched_slice = _saved_tensor_value(patched_logits)[0, :min_token_len, :].float()
                            kl_value = mean_sequence_kl(clean_logits_value, patched_slice).item()
                            total_kl[layer_idx, matrix_col] += kl_value
                            kl_counts[layer_idx, matrix_col] += 1

                            total_results[layer_idx, matrix_col] += patched_result_saved.item()
                            counts[layer_idx, matrix_col] += 1
    
    return total_results, counts, total_kl, kl_counts




def run_average_residual_stream_patching(dataset, N_LAYERS):
    
        total_results, counts, total_kl, kl_counts = average_residual_stream_patching(N_LAYERS, dataset)
        avg_results = np.divide(total_results, counts, out=np.zeros_like(total_results), where=counts!=0)
        avg_kl = np.divide(total_kl, kl_counts, out=np.zeros_like(total_kl), where=kl_counts != 0)
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

        plot_patching_kl_heatmap(
            avg_kl,
            labels,
            y_labels,
            "Dataset-Averaged KL(clean || patched), Residual Stream Patching",
            "figures/averaged_residual_stream_patching_kl_multiple.png",
            "figures/averaged_residual_stream_patching_kl_multiple.eps",
            xlabel="Token position (relative to ',')",
            ylabel="Layer",
        )

        plot_kl_vs_layer_line(
            avg_kl,
            "Dataset-Averaged KL(clean || patched), Residual Stream Patching",
            "figures/averaged_residual_stream_patching_kl_multiple_line.png",
            "figures/averaged_residual_stream_patching_kl_multiple_line.eps",
        )
        return plt.gcf()







def average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset):
    accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
    total_kl = np.zeros((N_LAYERS, N_HEADS))
    kl_counts = np.zeros((N_LAYERS, N_HEADS))
    
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
                clean_logits_value = _saved_tensor_value(clean_logits)[0, :min_token_len, :].float()
                np.save(f"data/clean_logits_value.npy", clean_logits_value.detach().cpu().numpy())
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
                
                for kl_layer_idx in range(N_LAYERS):
                    for kl_head_idx in range(N_HEADS):
                        patched_slice = _saved_tensor_value(
                            patched_logits_dict[(kl_layer_idx, kl_head_idx)]
                        ).float()
                        kl_value = mean_sequence_kl(clean_logits_value, patched_slice).item()
                        total_kl[kl_layer_idx, kl_head_idx] += kl_value
                        kl_counts[kl_layer_idx, kl_head_idx] += 1
        
    average_patching_results = accumulated_results / total_combinations
    avg_kl = np.divide(total_kl, kl_counts, out=np.zeros_like(total_kl), where=kl_counts != 0)
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
    return average_patching_results, avg_kl




def run_average_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS):
    average_patching_results, avg_kl = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = "averaged_attention_multiple"

    y_labels_attn = list(range(1, N_LAYERS + 1))
    plot_patching_kl_heatmap(
        avg_kl,
        x_labels,
        y_labels_attn,
        f"Dataset-Averaged KL(clean || patched), Attention Head Patching ({len(dataset)} idioms)",
        "figures/averaged_attention_head_patching_kl_multiple.png",
        "figures/averaged_attention_head_patching_kl_multiple.eps",
        xlabel="Attention head",
        ylabel="Layer",
    )
    plot_kl_vs_layer_line(
        avg_kl,
        f"Mean KL(clean || patched) vs layer — attention head patching ({len(dataset)} idioms)",
        "figures/averaged_attention_head_patching_kl_multiple_line.png",
        "figures/averaged_attention_head_patching_kl_multiple_line.eps",
    )
    return plt.gcf()




















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
        elif args.intervention == "attention_head":
            if args.averaging:
                run_average_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS)
        
    else:
        raise ValueError(f"Invalid model: {args.model}")
