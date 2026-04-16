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

import model_utils
import data_utils

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


def get_final_mod_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="final_modifier_extended", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=final_mod_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, save_attention=save_attention)

def get_first_head_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="first_head_extended", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=first_head_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, save_attention=save_attention)

def get_first_mod_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="first_modifier_extended", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=first_mod_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, save_attention=save_attention)

def get_final_head_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="final_head_extended", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=final_head_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, save_attention=save_attention)

def get_final_word_token_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers, load_if_available=True, batch_size = 1, rep_type="final_word_extended", rep_loc='./data', torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=final_word_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence_and": corrected_form_compounds_per_sentence_and}, save_attention=save_attention)



def get_noun_noun_compound_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers, load_if_available=True, batch_size = 1,  rep_loc='./data', rep_type="idiom_context", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=noun_noun_compound_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, 
     add_arg_dict={"corrected_form_compounds_per_sentence": corrected_form_compounds_per_sentence}, middle_dim=2, save_attention=save_attention)


def get_mean_pooled_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers, load_if_available=True, batch_size = 1, rep_loc='./data', rep_type="idiom_mean_pooled", torch_device="cuda", save_attention=False):
    return get_tokens_from_layers(data_loc=rep_loc, model_name=model_name, model=model, tokeniser=tokeniser, input_ids=input_ids, attention_mask=attention_mask, layers=layers,
     token_selector=mean_pool_selector, load_if_available=load_if_available, batch_size=batch_size, rep_type=rep_type, torch_device=torch_device, save_attention=save_attention)



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

def mean_pool_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i=None):
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    
    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024', 'FacebookAI/roberta-base', 'tohoku-nlp/bert-base-japanese']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B']:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    return np.vstack([np.mean(reps[get_tokens_to_keep(input_ids[i])].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])







def final_mod_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):
    
    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    # Get the modifier word tokens
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    mod_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
    mod_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in mod_word_input_ids_per_sent_raw]
    mod_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(mod_word_input_ids_per_sent)]

    # If there is only one modifier token, return the token, if there is more than one modifier token, return the last token
    mod_word_reps = np.vstack([reps[mod_word_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    return mod_word_reps

def final_head_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    # Get the head noun tokens
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
    head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]

    # If there is only one head noun token, return the token, if there is more than one head noun token, return the last token
    head_noun_reps = np.vstack([reps[head_noun_locs_per_sent[i][-1]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    return head_noun_reps

def first_mod_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):
    
    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    # Get the modifier word tokens
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    mod_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
    mod_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in mod_word_input_ids_per_sent_raw]
    mod_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(mod_word_input_ids_per_sent)]

    # If there is only one modifier token, return the token, if there is more than one modifier token, return the last token
    mod_word_reps = np.vstack([reps[mod_word_locs_per_sent[i][0]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    return mod_word_reps

def first_head_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    # Get the head noun tokens
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    # Get tokens where tokens aren't special tokens or pad tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
    head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]

    # If there is only one head noun token, return the token, if there is more than one head noun token, return the last token
    head_noun_reps = np.vstack([reps[head_noun_locs_per_sent[i][0]].cpu().numpy() for i, reps in enumerate(layer_reps)])
    return head_noun_reps

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



def noun_noun_compound_selector(model, model_name, tokeniser, token_reps, input_ids, layer, batch_size, i, corrected_form_compounds_per_sentence):
    # Get tokens where tokens aren't special tokens or pad tokens

    if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024', 'FacebookAI/roberta-base', 'tohoku-nlp/bert-base-japanese']:
        layer_reps = token_reps[1][layer].cpu()[:, :, :]
    elif layer == model.config.num_hidden_layers:
        layer_reps = token_reps[0].cpu()[:, :, :]
    elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base"]:
        layer_reps = token_reps[layer].cpu()[:, :, :]
    else:
        layer_reps = token_reps[2][layer].cpu()[:, :, :]
    
    compounds_per_batch = np.array([(' ' + x[0], ' ' + x[1]) for x in corrected_form_compounds_per_sentence])[i:i+batch_size]
    head_noun_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 1].tolist())['input_ids'])
    mod_word_input_ids_per_sent_raw = np.array(tokeniser.batch_encode_plus(compounds_per_batch[:, 0].tolist())['input_ids'])
    
    # Remove special tokens
    non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
    get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x)).reshape(-1)
    head_noun_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in head_noun_input_ids_per_sent_raw]
    mod_word_input_ids_per_sent = [np.array(x)[get_tokens_to_keep(x)] for x in mod_word_input_ids_per_sent_raw]

    # if model_name == 'xlm-mlm-xnli15-1024':
    #     head_noun_input_ids_per_sent = [x[1:] for x in head_noun_input_ids_per_sent]
    #     mod_word_input_ids_per_sent = [x[1:] for x in mod_word_input_ids_per_sent]

    head_noun_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(head_noun_input_ids_per_sent)]
    mod_word_locs_per_sent = [search_sequence_numpy(input_ids[i, :].cpu().numpy().reshape(-1), x.reshape(-1)) for i, x in enumerate(mod_word_input_ids_per_sent)]

    head_noun_reps = np.vstack([np.mean(reps[head_noun_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
    mod_word_reps = np.vstack([np.mean(reps[mod_word_locs_per_sent[i]].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
    compound_reps = np.stack([mod_word_reps, head_noun_reps], axis=1)
    
    # Shape = (batch_size, 2, hidden_size)
    return compound_reps


def get_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers, token_selector=mean_pool_selector, load_if_available=True, batch_size = 1, rep_type="sentence_pair_cls", torch_device="cuda", save_reps=True, data_loc='./data', add_arg_dict={}, middle_dim=None, save_attention=False):
    '''
    layer 0: embedding
    layer 1: first layer
    . . . 
    layer 12: usually the final layer
    NOTE: Would probably be much faster to extract all layers at the same time (assuming we want outputs from all layers)
    '''
    rep_locs_per_layer = [data_utils.get_hidden_state_file(model_name, layer=x, rep_type=rep_type, data_loc=data_loc) for x in layers]
    load_reps = load_if_available and os.path.isfile(rep_loc)

    layer_map = dict(zip(layers, range(len(layers))))

    if load_reps:
        tokens_per_layer = [np.load(x) for x in rep_locs_per_layer]
    else:
        print('Extracting representations from model for layers {}'.format(layers))
        
        input_ids = input_ids.to(torch_device)
        attention_mask = attention_mask.to(torch_device)
        model.to(torch_device)
        
        # TODO: Work this out from model object
        embedding_size = 1024 if 'xlm' in model_name else 768
        tokens_per_layer = [np.zeros((input_ids.shape[0], embedding_size)) if middle_dim == None else np.zeros((input_ids.shape[0], middle_dim, embedding_size)) for x in layers]

        if save_attention:
            # Calculate size of attention matrix
            non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
            pad_token_mask = lambda x: np.array(x.cpu() == tokeniser.pad_token_id)
            get_tokens_to_keep = lambda x: np.argwhere((pad_token_mask(x) == False)).reshape(-1)
            tokens_to_keep_per_sample = [get_tokens_to_keep(x) for x in input_ids]
            seq_lens = np.array([len(x) for x in tokens_to_keep_per_sample])
            seq_len = max(seq_lens)
            attention_per_layer = [-np.ones((input_ids.shape[0], seq_len**2)) for x in layers]

            decode_tokens = lambda x: [tokeniser.decode(token) for token in x[get_tokens_to_keep(x)].tolist()]
            '\n'.join([''.join(decode_tokens(x)) for x in input_ids])

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
                token_reps = model(input_ids[i:i+batch_size, :].reshape(batch_size, -1), attention_mask=attention_mask[i:i+batch_size, :].reshape(batch_size, -1), output_attentions=save_attention)
                
                add_arg_dict["i"] = i

                if save_attention:
                    for layer in layers:
                        tokens_per_layer[layer_map[layer]][i:i+batch_size, :] = token_selector(model, model_name, tokeniser, token_reps[:-1], input_ids[i:i+batch_size, :].reshape(batch_size, -1), layer, batch_size, **add_arg_dict)
                        for batch_offset, sample_attention in enumerate(token_reps[-1][layer - 1]):
                            sample_i = i + batch_offset
                            sample_len = seq_lens[sample_i]
                            attention_per_layer[layer_map[layer]][sample_i, :sample_len**2] = sample_attention.mean(axis=0)[tokens_to_keep_per_sample[sample_i]][:, tokens_to_keep_per_sample[sample_i]].reshape(sample_len**2).cpu()
                else:
                    for layer in layers:
                       tokens_per_layer[layer_map[layer]][i:i+batch_size, :] = token_selector(model, model_name, tokeniser, token_reps, input_ids[i:i+batch_size, :].reshape(batch_size, -1), layer, batch_size, **add_arg_dict)
        
        if save_reps:
            for layer in layers:
                pathlib.Path('/'.join(rep_locs_per_layer[layer_map[layer]].split('/')[:-1])).mkdir(parents=True, exist_ok=True)
                np.save(rep_locs_per_layer[layer_map[layer]], tokens_per_layer[layer_map[layer]])
    
    if save_attention:
        for layer in layers:
            atten_rep_loc = data_utils.get_hidden_state_file(model_name, layer=layer, rep_type=rep_type + '_attention', data_loc=data_loc)
            pathlib.Path('/'.join(atten_rep_loc.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
            np.save(atten_rep_loc, attention_per_layer[layer_map[layer]])

        return tokens_per_layer, attention_per_layer
    else:
        return tokens_per_layer



def extract_and_save_representations(amount_of_dataset=1, batch_size=1, models=None, load_if_available=False, layers=None, torch_device="cpu", representations=["mean_pooled"], rep_loc='./data', save_attention=False):

    if representations == None:
        representations = ["mean_pooled", "idiom_context"]

   
    sentences = data_utils.get_extended_sentences()
    mod_head_words_per_sentence_with_context = data_utils.get_idiom_modifier_head_words_per_sentence_extended()
    head_and_words_per_sentence_with_context = data_utils.get_idiom_modifier_head_words_per_sentence_extended_and()
    num_to_keep = int(amount_of_dataset * len(sentences))
    sentences = sentences[:num_to_keep]
    mod_head_words_per_sentence_with_context = mod_head_words_per_sentence_with_context[:num_to_keep]
    head_and_words_per_sentence_with_context = head_and_words_per_sentence_with_context[:num_to_keep]

   
    initial_layers = layers

    if models == None:
        models = model_utils.dev_model_configs.keys()

    for model_name in tqdm.tqdm(models):
        print('Loading {}'.format(model_name))
        model, tokeniser = model_utils.load_model(model_name)
        if tokeniser.pad_token is None:
            if tokeniser.eos_token:
                tokeniser.pad_token = tokeniser.eos_token
            else:
                tokeniser.add_special_tokens({'pad_token': '<pad>'})

        unpack_dict = lambda x: (x['input_ids'], x['attention_mask'])

        if initial_layers == None:
            layers = range(1, model.config.num_hidden_layers + 1)

        layers = [x for x in layers if x in range(1, model.config.num_hidden_layers + 1)]
        
       
        if "final_modifier_extended" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_correct_form_extended()
            get_final_mod_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_modifier_extended', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        
        if "final_head_extended" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_correct_form_extended()
            get_final_head_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_head_extended', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        
        if "final_word_extended" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence_and = data_utils.load_correct_form_extended_and()
            get_final_word_token_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence_and, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='final_word_extended', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        
        if "first_modifier_extended" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_correct_form_extended()
            get_first_mod_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='first_modifier_extended', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        
        if "first_head_extended" in representations:
            inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            #input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_correct_form_extended()
            get_first_head_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='first_head_extended', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
        



        if "mean_pooled" in representations:
            # inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            # input_ids = inputs["input_ids"]
            # attention_mask = inputs["attention_mask"]
            input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            get_mean_pooled_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='idiom_mean_pooled', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)


        if "idiom_context" in representations:
            # inputs = tokeniser(sentences.tolist(),  max_length = 512, return_tensors="pt", truncation=True, pad_to_max_length=True)
            # input_ids = inputs["input_ids"]
            # attention_mask = inputs["attention_mask"]
            input_ids, attention_mask = unpack_dict(tokeniser.batch_encode_plus(sentences.tolist(), max_length=512, return_tensors='pt', pad_to_max_length=True))
            corrected_form_compounds_per_sentence = data_utils.load_correct_form_extended()
            get_noun_noun_compound_tokens_from_layers(model_name, model, tokeniser, input_ids, attention_mask, corrected_form_compounds_per_sentence, layers=layers, load_if_available=load_if_available, batch_size=batch_size, rep_type='idiom_context', rep_loc=rep_loc, torch_device=torch_device, save_attention=save_attention)
       

       


        else:
            print('No representations specified')

if __name__ == "__main__":
    args = parser.parse_args()
    print(vars(args))

    extract_and_save_representations(amount_of_dataset=args.amount_of_dataset, load_if_available=args.load_if_available, batch_size=args.batch_size, models=args.models, layers=args.layers, torch_device=args.device, representations=args.representations, rep_loc=args.rep_loc, save_attention=args.save_attention)
