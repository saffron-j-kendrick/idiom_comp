## IMPORTS

import os
import pickle
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
from scipy.stats import ttest_ind, ttest_rel, wilcoxon
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
from itertools import product
import pandas as pd
import xlsxwriter
from matplotlib.lines import Line2D
import rsa_utils
import data_utils



## DATA

df = pd.read_excel("data/standard_sentences.xlsx")
#corr_metric = 'kendalltau'
corr_metric = 'spearmanr'

order_dict = dict(zip(["openai-community/gpt2", 'mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen2.5-7B' ], range(0, 6)))
model_name_map = {'mistralai/Mistral-7B-v0.1': 'Mistral-7B', 'meta-llama/Llama-3.2-3B' : 'Llama-3.2-3B', "tiiuae/Falcon3-7B-Base" : "Falcon3-7B", "openai-community/gpt2" : "GPT2",'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' : 'DeepSeek-R1-1.5B', 'Qwen/Qwen2.5-7B' : 'Qwen2.5-7B'}

# order_dict = dict(zip(["meta-llama/Llama-3.2-3B"], range(0, 1)))
# model_name_map = {'meta-llama/Llama-3.2-3B' : 'Llama-3.2-3B'}

def sort_df_by_model_order(df, keep_order_col=True, update_names=True):
    df['model_order'] = [order_dict[x.model] for x in df.iloc]

    extra_columns_to_sort = ['representation', 'Word representations processed . . .']
    extra_columns_to_sort = [x for x in extra_columns_to_sort if x in df.columns]

    sort_cols = ['model_order'] + extra_columns_to_sort
    df = df.sort_values(sort_cols)

    if not keep_order_col:
        del df['model_order']

    if update_names:
        df['model_name'] = [model_name_map[x.model] for x in df.iloc]
        if 'representation' in df.columns:
            df['representation_name'] = [model_name_map[x.representation] if x.representation in model_name_map else x.representation for x in df.iloc]

    return df


def get_significant_layers_for_panel(
    no_ablation_df,
    idiomaticity_ablation_df,
    representation_name,
    metric,
    test_kind='ttest',
    alpha=0.05,
):
    """Return FDR-corrected significant layers for no-ablation vs idiomaticity-ablation."""
    key_cols = ['model', 'layer', 'Representation', metric]
    no_df = no_ablation_df[no_ablation_df['Representation'] == representation_name][key_cols].copy()
    ablation_df = idiomaticity_ablation_df[idiomaticity_ablation_df['Representation'] == representation_name][key_cols].copy()

    no_df = no_df.sort_values(['model', 'layer']).reset_index(drop=True)
    ablation_df = ablation_df.sort_values(['model', 'layer']).reset_index(drop=True)

    # Pair repeated rows deterministically so paired tests are valid.
    no_df['pair_idx'] = no_df.groupby(['model', 'layer', 'Representation']).cumcount()
    ablation_df['pair_idx'] = ablation_df.groupby(['model', 'layer', 'Representation']).cumcount()

    merged = no_df.merge(
        ablation_df,
        on=['model', 'layer', 'Representation', 'pair_idx'],
        suffixes=('_no', '_ablation'),
    )
    if merged.empty:
        return set()

    rows = []
    for layer, layer_df in merged.groupby('layer'):
        vals_no = layer_df[f'{metric}_no'].astype(float).to_numpy()
        vals_ablation = layer_df[f'{metric}_ablation'].astype(float).to_numpy()
        valid = (~np.isnan(vals_no)) & (~np.isnan(vals_ablation))
        vals_no = vals_no[valid]
        vals_ablation = vals_ablation[valid]
        if len(vals_no) < 2:
            continue
        try:
            # Match idiom_rsa_complete.py: one-sided paired tests.
            if test_kind == 'wilcoxon':
                p_val = wilcoxon(vals_no, vals_ablation, alternative='greater').pvalue
            else:
                # Avoid noisy precision-loss warnings when vectors are effectively identical.
                if np.allclose(vals_no, vals_ablation, rtol=1e-12, atol=1e-12):
                    p_val = 1.0
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore',
                            message='Precision loss occurred in moment calculation due to catastrophic cancellation.*',
                            category=RuntimeWarning,
                        )
                        p_val = ttest_rel(vals_no, vals_ablation, alternative='greater').pvalue
        except ValueError:
            p_val = np.nan
        rows.append({'layer': layer, 'p': p_val})

    if not rows:
        return set()

    p_df = pd.DataFrame(rows)
    p_df['corrected_p'] = np.nan
    valid_mask = p_df['p'].notna()
    if valid_mask.any():
        # Match idiom_rsa_complete.py correction settings.
        p_df.loc[valid_mask, 'corrected_p'] = fdrcorrection(
            p_df.loc[valid_mask, 'p'].tolist(),
            method='indep',
            alpha=alpha,
        )[1]

    return set(p_df.loc[p_df['corrected_p'] < alpha, 'layer'].tolist())


def add_significance_markers(ax, panel_df, metric, significant_layers):
    if not significant_layers:
        return
    for layer in sorted(significant_layers):
        ax.text(layer, 0.0, '*', ha='center', va='bottom', fontsize=12, color='black')


def configure_panel_legend(ax, panel_index):
    legend = ax.get_legend()
    if legend is None:
        return
    if panel_index == 0:
        handles, labels = ax.get_legend_handles_labels()
        if 'Significant difference' not in labels:
            handles.append(
                Line2D([], [], marker='*', linestyle='None', color='black', markersize=10)
            )
            labels.append('Significant difference')
        legend.remove()
        legend = ax.legend(handles=handles, labels=labels, loc='upper left', title='')
        legend.set_bbox_to_anchor((0.02, 0.98))
    else:
        legend.remove()


sentences = np.array(df['sentence'].tolist())
df['expression'] = ['{} {}'.format(x['verb'], x['noun']) for x in df.iloc]
idioms = np.array(df['idiom'].tolist())
words_per_sent = [x.split(' ') for x in sentences]
words_per_sent = [[x.strip("'.,!?") for x in sent] for sent in words_per_sent]
words_per_sent = [[x.replace("'", "") for x in sent] for sent in words_per_sent]
words_per_sent = [[x.replace("-", "") for x in sent] for sent in words_per_sent]
process_sent = lambda x: [y for y in nltk.word_tokenize(x.strip().lower()) if y.isalpha()]
# lemmatiser = WordNetLemmatizer()
# word_dict = {'gestates' : 'gestate'}
# look_up = lambda word: word_dict[word] if word in word_dict else lemmatiser.lemmatize(word)
# get_vector = lambda word: fasttext[word] if word in fasttext else fasttext[look_up(word)] if word.strip() else np.zeros(fasttext.vector_size)

# def get_average_vector(words):
#     return np.vstack([get_vector(x) for x in words]).mean(axis=0)


# load = False

# if not load:
#     fasttext = gensim.models.KeyedVectors.load_word2vec_format('wiki.en.vec', limit=500000)
#     mean_fasttext_reps_per_sent = np.vstack([get_average_vector(x) for x in words_per_sent])
#     np.save('results/mean_fasttext_reps.npy', mean_fasttext_reps_per_sent)
# else:
#     mean_fasttext_reps_per_sent = np.load('results/mean_fasttext_reps.npy')

# identity_rdm = np.ones((200, 1200))

# identity_rdm[np.arange(200), np.arange(200)] = 0


# fasttext_mean_rdm = rsa_utils.get_rdm(mean_fasttext_reps_per_sent, 'cosine')
# rsa_utils.plot_mtx(fasttext_mean_rdm, 'FastText')




model_names = ['meta-llama/Llama-3.2-3B','openai-community/gpt2', 'mistralai/Mistral-7B-v0.1', 'tiiuae/Falcon3-7B-Base', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen2.5-7B']

GROUP_TO_SAVE = 27  # saves one 8x8 group instead of full 320x320 RDM
## RDMS
phrases = np.array(df['expression'].tolist())

labels_within_group = np.array([
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1]
])
same_relation_group_rdm = rsa_utils.get_rdm(labels_within_group)
rsa_utils.plot_mtx(same_relation_group_rdm, "")

current_cmap = plt.get_cmap('Spectral_r')
current_cmap.set_bad(color='gray')

start = 27
sent_inds = np.arange(start * 8, start * 8 + 8)

fig, ax = plt.subplots(figsize=(6, 4))
# plt.rcParams.update({'font.size': 12})
figure_rdm = same_relation_group_rdm
plt.imshow(same_relation_group_rdm, interpolation='nearest', cmap=current_cmap)

cb = plt.colorbar(label='Dissimilarity')
labels = np.arange(0, 1)
cb.set_ticks(labels)
#cb.set_ticklabels(np.arange(11) / 10)
plt.xticks(ticks=np.arange(8), labels=['{}'.format(phrases[x], x) for x in sent_inds], rotation=45, ha='right');
plt.yticks(ticks=np.arange(8), labels=['{}'.format(phrases[x], x) for x in sent_inds], rotation=0, ha='right');
ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

plt.tight_layout()
# plt.savefig('figures/same_literal_meaning_ground_truth_rdm.png', bbox_inches='tight')
# plt.savefig('figures/same_literal_meaning_ground_truth_rdm.eps', format='eps', bbox_inches='tight')


same_head_rdm = np.zeros((len(phrases), len(phrases)))

data = []
for compound in phrases:
    # Split each phrase into words and add as a separate "sentence"
    data.append(compound.split())

skip_gram = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)




for i, compound_i in enumerate(phrases):
    for j, compound_j in enumerate(phrases):
        same_head_rdm[i, j] = 1 - skip_gram.wv.similarity(compound_i.split()[-1], compound_j.split()[-1])


start = 27
sent_inds = np.arange(start * 8, start * 8 + 8)


fig, ax = plt.subplots(figsize=(8, 6))
plt.imshow(same_head_rdm[sent_inds, :][:, sent_inds], interpolation='nearest', cmap='Spectral_r')
plt.title('')
plt.colorbar();

plt.xticks(ticks=np.arange(8), labels=['{}'.format(phrases[x], x) for x in sent_inds], rotation=45, ha='right');
plt.yticks(ticks=np.arange(8), labels=['{}'.format(phrases[x], x) for x in sent_inds], rotation=0, ha='right');

plt.tight_layout()
# plt.savefig('figures/same_noun_adj_ground_truth_rdm.png', bbox_inches='tight')
# plt.savefig('figures/same_noun_adj_ground_truth_rdm.eps', format='eps', bbox_inches='tight')



same_modifier_rdm = np.zeros((len(phrases), len(phrases)))

data = []
for compound in phrases:
    # Split each phrase into words and add as a separate "sentence"
    data.append(compound.split())

skip_gram = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)

same_modifier_rdm = np.zeros((len(phrases), len(phrases)))

for i, compound_i in enumerate(phrases):
    for j, compound_j in enumerate(phrases):
        same_modifier_rdm[i, j] = 1 - skip_gram.wv.similarity(compound_i.split()[0], compound_j.split()[0])


start = 27
sent_inds = np.arange(start * 8, start * 8 + 8)
# rdm_inds = np.arange(0, 15, 3)

fig, ax = plt.subplots(figsize=(8, 6))
plt.imshow(same_modifier_rdm[sent_inds, :][:, sent_inds], interpolation='nearest', cmap='Spectral_r')
plt.title('')
plt.colorbar();

plt.xticks(ticks=np.arange(8), labels=['{}'.format(phrases[x], x) for x in sent_inds], rotation=45, ha='right');
plt.yticks(ticks=np.arange(8), labels=['{}'.format(phrases[x], x) for x in sent_inds], rotation=0, ha='right');

plt.tight_layout()
# plt.savefig('figures/same_verb_ground_truth_rdm.png', bbox_inches='tight')
# plt.savefig('figures/same_verb_ground_truth_rdm.eps', format='eps', bbox_inches='tight')

group_rdms_to_correlate = [("same_relation_group_rdm", same_relation_group_rdm), ("same_head_rdm", same_head_rdm), ("same_modifier_rdm", same_modifier_rdm)]


def corr_within_group(rdm_a, rdm_b):
    return corr(data_utils.select_within_compound_groups(rdm_a.reshape(320, 8)), data_utils.select_within_compound_groups(rdm_b.reshape(320, 8)))



###

model_names = ['meta-llama/Llama-3.2-3B']

load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_standard_attention_head_masked_168_mlp_masked_168/{}_layer_{}_final_standard_attention_head_masked_168_mlp_masked_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_context_attention_head_masked_168_mlp_masked_168/{}_layer_{}_final_context_attention_head_masked_168_mlp_masked_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_literal_attention_head_masked_168_mlp_masked_168/{}_layer_{}_final_literal_attention_head_masked_168_mlp_masked_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_standard_attention_head_masked_168_mlp_masked_168_random_run{}/'
                                '{}_layer_{}_final_standard_attention_head_masked_168_mlp_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_context_attention_head_masked_168_mlp_masked_168_random_run{}/'
                                '{}_layer_{}_final_context_attention_head_masked_168_mlp_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_literal_attention_head_masked_168_mlp_masked_168_random_run{}/'
                                '{}_layer_{}_final_literal_attention_head_masked_168_mlp_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure:
# panel 1 = standard, panel 2 = context, panel 3 = no context
# each panel overlays normal vs ablation vs random ablation.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Llama 3.2 Attention Head and MLP Ablation 25%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_168_mlp_168_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_168_mlp_168_with_mask.eps', format='eps')
plt.show()









load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_standard_attention_head_masked_67_mlp_masked_67/{}_layer_{}_final_standard_attention_head_masked_67_mlp_masked_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_context_attention_head_masked_67_mlp_masked_67/{}_layer_{}_final_context_attention_head_masked_67_mlp_masked_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_literal_attention_head_masked_67_mlp_masked_67/{}_layer_{}_final_literal_attention_head_masked_67_mlp_masked_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_standard_attention_head_masked_67_mlp_masked_67_random_run{}/'
                                '{}_layer_{}_final_standard_attention_head_masked_67_mlp_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_context_attention_head_masked_67_mlp_masked_67_random_run{}/'
                                '{}_layer_{}_final_context_attention_head_masked_67_mlp_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_literal_attention_head_masked_67_mlp_masked_67_random_run{}/'
                                '{}_layer_{}_final_literal_attention_head_masked_67_mlp_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure:
# panel 1 = standard, panel 2 = context, panel 3 = no context
# each panel overlays normal vs ablation vs random ablation.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Llama 3.2 Attention Head and MLP Ablation 10%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_67_mlp_67_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_67_mlp_67_with_mask.eps', format='eps')
plt.show()







###

model_names = ['meta-llama/Llama-3.2-3B']

load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_67/{}_layer_{}_final_word_standard_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_67/{}_layer_{}_final_word_context_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_67/{}_layer_{}_final_word_literal_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Llama 3.2 Attention Head Ablation 10%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_67_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_67_with_mask.eps', format='eps')
plt.show()


load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_168/{}_layer_{}_final_word_standard_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_168/{}_layer_{}_final_word_context_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_168/{}_layer_{}_final_word_literal_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Llama 3.2 Attention Head Ablation 25%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_168_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_168_with_mask.eps', format='eps')
plt.show()





### with full region



load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_standard_attention_head_masked_168_mlp_masked_168/{}_layer_{}_final_standard_attention_head_masked_168_mlp_masked_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_context_attention_head_masked_168_mlp_masked_168/{}_layer_{}_final_context_attention_head_masked_168_mlp_masked_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_literal_attention_head_masked_168_mlp_masked_168/{}_layer_{}_final_literal_attention_head_masked_168_mlp_masked_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_standard_attention_head_masked_168_mlp_masked_168_random_run{}/'
                                '{}_layer_{}_final_standard_attention_head_masked_168_mlp_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_context_attention_head_masked_168_mlp_masked_168_random_run{}/'
                                '{}_layer_{}_final_context_attention_head_masked_168_mlp_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_literal_attention_head_masked_168_mlp_masked_168_random_run{}/'
                                '{}_layer_{}_final_literal_attention_head_masked_168_mlp_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure:
# panel 1 = standard, panel 2 = context, panel 3 = no context
# each panel overlays normal vs ablation vs random ablation.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Llama 3.2 Attention Head and MLP Ablation 25%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_168_mlp_168.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_168_mlp_168.eps', format='eps')
plt.show()









load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_standard_attention_head_masked_67_mlp_masked_67/{}_layer_{}_final_standard_attention_head_masked_67_mlp_masked_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_context_attention_head_masked_67_mlp_masked_67/{}_layer_{}_final_context_attention_head_masked_67_mlp_masked_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_literal_attention_head_masked_67_mlp_masked_67/{}_layer_{}_final_literal_attention_head_masked_67_mlp_masked_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_standard_attention_head_masked_67_mlp_masked_67_random_run{}/'
                                '{}_layer_{}_final_standard_attention_head_masked_67_mlp_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_context_attention_head_masked_67_mlp_masked_67_random_run{}/'
                                '{}_layer_{}_final_context_attention_head_masked_67_mlp_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_literal_attention_head_masked_67_mlp_masked_67_random_run{}/'
                                '{}_layer_{}_final_literal_attention_head_masked_67_mlp_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure:
# panel 1 = standard, panel 2 = context, panel 3 = no context
# each panel overlays normal vs ablation vs random ablation.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Llama 3.2 Attention Head and MLP Ablation 10%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_67_mlp_67.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_67_mlp_67.eps', format='eps')
plt.show()







###

model_names = ['meta-llama/Llama-3.2-3B']

load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_67/{}_layer_{}_final_word_standard_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_67/{}_layer_{}_final_word_context_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_67/{}_layer_{}_final_word_literal_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Llama 3.2 Attention Head Ablation 10%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_67.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_67.eps', format='eps')
plt.show()


load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_168/{}_layer_{}_final_word_standard_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_168/{}_layer_{}_final_word_context_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_168/{}_layer_{}_final_word_literal_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Llama 3.2 Attention Head Ablation 25%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_168.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_168.eps', format='eps')
plt.show()



## need to add a signifcance test to the figure but use the wilcoxon test for the outlined region as this is non parametric. change the FDR to 'indep'


def format_p_val(p):

    if hasattr(p, 'pvalue'):
        p = p.pvalue
    p = round(p, 2)
    
    if p == -1:
        p = '-'
    elif p < 0.0001:
        p = 'p<0.0001'  
    elif p < 0.001:
        p = 'p<0.001'
    elif p < 0.01:
        p = 'p<0.01'
    elif p < 0.05:
        p = 'p<0.05'
    else:
        p = 'p={}'.format(p)
    
    return p 

def bonferonni_correction(df):
    # p == -1 for distilroberta layers 7-12 (i.e. non-existent layers)
    fdr_method = 'indep' # 'indep' or 'negcorr
    num_tests = paired_t_test_df[paired_t_test_df.p != -1].shape[0]
    
    df.loc[df.p != -1, 'corrected_p'] =  paired_t_test_df[paired_t_test_df.p != -1].p * num_tests
    df.loc[df.p == -1, 'corrected_p'] = -1

    df['formatted_corrected_p']  = list(map(format_p_val, df.corrected_p))
    
    return df

def fdr_correction(df):
    # p == -1 for distilroberta layers 7-12 (i.e. non-existent layers)
    fdr_error_rate = 0.05 # default = 0.05
    fdr_method = 'indep' # 'indep' or 'negcorr
    df.loc[df.p > -1, 'corrected_p'] = fdrcorrection(df[df.p > -1].p.tolist(), method=fdr_method, alpha=fdr_error_rate)[-1]
    df.loc[df.p == -1, 'corrected_p'] = -1
    df['formatted_corrected_p']  = list(map(format_p_val, df.corrected_p))
    
    return df


###

model_names = ["openai-community/gpt2"]

load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_14/{}_layer_{}_final_word_standard_attention_head_masked_significant_14.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_14/{}_layer_{}_final_word_context_attention_head_masked_significant_14.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_14/{}_layer_{}_final_word_literal_attention_head_masked_significant_14.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_14_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_14_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_14_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_14_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_14_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_14_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('GPT Attention Head Ablation 10%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_14.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_14.eps', format='eps')
plt.show()





load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_14/{}_layer_{}_final_word_standard_attention_head_masked_significant_14.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_14/{}_layer_{}_final_word_context_attention_head_masked_significant_14.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_14/{}_layer_{}_final_word_literal_attention_head_masked_significant_14.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_14_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_14_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_14_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_14_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_14_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_14_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('GPT Attention Head Ablation 10%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_14_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_14_with_mask.eps', format='eps')
plt.show()




load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_36/{}_layer_{}_final_word_standard_attention_head_masked_significant_36.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_36/{}_layer_{}_final_word_context_attention_head_masked_significant_36.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_36/{}_layer_{}_final_word_literal_attention_head_masked_significant_36.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_36_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_36_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_36_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_36_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_36_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_36_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('GPT Attention Head Ablation 25%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_36.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_36.eps', format='eps')
plt.show()





load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_36/{}_layer_{}_final_word_standard_attention_head_masked_significant_36.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_36/{}_layer_{}_final_word_context_attention_head_masked_significant_36.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_36/{}_layer_{}_final_word_literal_attention_head_masked_significant_36.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_36_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_36_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_36_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_36_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_36_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_36_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('GPT Attention Head Ablation 25%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_36_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_36_with_mask.eps', format='eps')
plt.show()





# ### 

# model_names = ["Qwen/Qwen2.5-7B"]


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
                       
#                         reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
#                     elif rep == 'context':
                        
#                         reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


#                     rdm = rsa_utils.get_rdm(reps)
                
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)
#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
                       
#                         reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_78/{}_layer_{}_final_word_standard_attention_head_masked_significant_78.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
#                     elif rep == 'context':
                        
#                         reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_78/{}_layer_{}_final_word_context_attention_head_masked_significant_78.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_78/{}_layer_{}_final_word_literal_attention_head_masked_significant_78.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


#                     rdm = rsa_utils.get_rdm(reps)
#                     # if layer==15:
#                     #     if model_name == 'meta-llama/Llama-3.2-3B':
#                     #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
#                     #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
#                     #         plt.close()
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)
#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_78_random_run{}/'
#                                 '{}_layer_{}_final_word_standard_attention_head_masked_78_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
#                     elif rep == 'context':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_context_attention_head_masked_78_random_run{}/'
#                                 '{}_layer_{}_final_word_context_attention_head_masked_78_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_78_random_run{}/'
#                                 '{}_layer_{}_final_word_literal_attention_head_masked_78_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

#                     rdm = rsa_utils.get_rdm(reps)
#                     # if layer==15:
#                     #     if model_name == 'meta-llama/Llama-3.2-3B':
#                     #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
#                     #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
#                     #         plt.close()
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)

#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# # 3-panel comparison figure: normal vs ablation vs random ablation
# # Each panel overlays standard/context/no_context curves.
# normal_df = pd.read_csv('results/idiom_representations_normal.csv')
# ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
# random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# # Keep plotting consistent with the selected model(s) in this run.
# selected_models = set(model_names)
# normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
# ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
# random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# # create directory for figures if it doesn't exist
# os.makedirs('figures', exist_ok=True)

# label_map = {
#     'standard': 'Neutral',
#     'context': 'Figurative',
#     'no_context': 'Literal',
# }

# condition_palette = {
#     'No Ablation': '#8E44AD',
#     'Idiomaticity Ablation': '#A6761D',
#     'Random Ablation': '#D81B60',
# }

# for df in (normal_df, ablation_df, random_ablation_df):
#     df['Representation'] = df['representation'].map(label_map)
#     # drop any rows with unexpected representation labels
#     df.dropna(subset=['Representation'], inplace=True)

# normal_df['Condition'] = 'No Ablation'
# ablation_df['Condition'] = 'Idiomaticity Ablation'
# random_ablation_df['Condition'] = 'Random Ablation'

# plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

# metric = 'same_relation_group_rdm_corr'
# fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
# fig.suptitle('Qwen Attention Head Ablation 10%, full region')
# panel_order = ['Neutral', 'Figurative', 'Literal']
# region_test = 'ttest'

# for i, rep_name in enumerate(panel_order):
#     ax = axes[i]
#     panel_df = plot_df[plot_df['Representation'] == rep_name]
#     significant_layers = get_significant_layers_for_panel(
#         normal_df,
#         ablation_df,
#         rep_name,
#         metric,
#         test_kind=region_test,
#     )
#     sns.lineplot(
#         data=panel_df,
#         x='layer',
#         y=metric,
#         hue='Condition',
#         style='Condition',
#         markers=True,
#         dashes=False,
#         palette=condition_palette,
#         ax=ax,
#     )
#     add_significance_markers(ax, panel_df, metric, significant_layers)
#     ax.set_title(rep_name)
#     ax.set_xlabel('Layer')
#     if i == 0:
#         ax.set_ylabel('Correlation')
#     else:
#         ax.set_ylabel('')
#     ax.axhline(0, color='black', linestyle='--', linewidth=1)
#     ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

#     configure_panel_legend(ax, i)

# fig.tight_layout(rect=[0, 0, 1, 0.93])
# plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_qwen_78.png', format='png')
# plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_qwen_78.eps', format='eps')
# plt.show()





# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
                       
#                         reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
#                     elif rep == 'context':
                        
#                         reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


#                     rdm = rsa_utils.get_rdm(reps)
                
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)
#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
                       
#                         reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_78/{}_layer_{}_final_word_standard_attention_head_masked_significant_78.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
#                     elif rep == 'context':
                        
#                         reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_78/{}_layer_{}_final_word_context_attention_head_masked_significant_78.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_78/{}_layer_{}_final_word_literal_attention_head_masked_significant_78.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


#                     rdm = rsa_utils.get_rdm(reps)
#                     # if layer==15:
#                     #     if model_name == 'meta-llama/Llama-3.2-3B':
#                     #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
#                     #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
#                     #         plt.close()
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)
#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_78_random_run{}/'
#                                 '{}_layer_{}_final_word_standard_attention_head_masked_78_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
#                     elif rep == 'context':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_context_attention_head_masked_78_random_run{}/'
#                                 '{}_layer_{}_final_word_context_attention_head_masked_78_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_78_random_run{}/'
#                                 '{}_layer_{}_final_word_literal_attention_head_masked_78_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

#                     rdm = rsa_utils.get_rdm(reps)
#                     # if layer==15:
#                     #     if model_name == 'meta-llama/Llama-3.2-3B':
#                     #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
#                     #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
#                     #         plt.close()
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)

#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# # 3-panel comparison figure: normal vs ablation vs random ablation
# # Each panel overlays standard/context/no_context curves.
# normal_df = pd.read_csv('results/idiom_representations_normal.csv')
# ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
# random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# # Keep plotting consistent with the selected model(s) in this run.
# selected_models = set(model_names)
# normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
# ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
# random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# # create directory for figures if it doesn't exist
# os.makedirs('figures', exist_ok=True)

# label_map = {
#     'standard': 'Neutral',
#     'context': 'Figurative',
#     'no_context': 'Literal',
# }

# condition_palette = {
#     'No Ablation': '#8E44AD',
#     'Idiomaticity Ablation': '#A6761D',
#     'Random Ablation': '#D81B60',
# }

# for df in (normal_df, ablation_df, random_ablation_df):
#     df['Representation'] = df['representation'].map(label_map)
#     # drop any rows with unexpected representation labels
#     df.dropna(subset=['Representation'], inplace=True)

# normal_df['Condition'] = 'No Ablation'
# ablation_df['Condition'] = 'Idiomaticity Ablation'
# random_ablation_df['Condition'] = 'Random Ablation'

# plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

# metric = 'same_relation_group_rdm_corr'
# fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
# fig.suptitle('Qwen Attention Head Ablation 10%, outlined region')
# panel_order = ['Neutral', 'Figurative', 'Literal']
# region_test = 'wilcoxon'

# for i, rep_name in enumerate(panel_order):
#     ax = axes[i]
#     panel_df = plot_df[plot_df['Representation'] == rep_name]
#     significant_layers = get_significant_layers_for_panel(
#         normal_df,
#         ablation_df,
#         rep_name,
#         metric,
#         test_kind=region_test,
#     )
#     sns.lineplot(
#         data=panel_df,
#         x='layer',
#         y=metric,
#         hue='Condition',
#         style='Condition',
#         markers=True,
#         dashes=False,
#         palette=condition_palette,
#         ax=ax,
#     )
#     add_significance_markers(ax, panel_df, metric, significant_layers)
#     ax.set_title(rep_name)
#     ax.set_xlabel('Layer')
#     if i == 0:
#         ax.set_ylabel('Correlation')
#     else:
#         ax.set_ylabel('')
#     ax.axhline(0, color='black', linestyle='--', linewidth=1)
#     ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

#     configure_panel_legend(ax, i)

# fig.tight_layout(rect=[0, 0, 1, 0.93])
# plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_qwen_78_with_mask.png', format='png')
# plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_qwen_78_with_mask.eps', format='eps')
# plt.show()




# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
                       
#                         reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
#                     elif rep == 'context':
                        
#                         reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


#                     rdm = rsa_utils.get_rdm(reps)
                
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)
#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
                       
#                         reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_196/{}_layer_{}_final_word_standard_attention_head_masked_significant_196.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
#                     elif rep == 'context':
                        
#                         reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_196/{}_layer_{}_final_word_context_attention_head_masked_significant_196.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_196/{}_layer_{}_final_word_literal_attention_head_masked_significant_196.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


#                     rdm = rsa_utils.get_rdm(reps)
#                     # if layer==15:
#                     #     if model_name == 'meta-llama/Llama-3.2-3B':
#                     #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
#                     #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
#                     #         plt.close()
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)
#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_196_random_run{}/'
#                                 '{}_layer_{}_final_word_standard_attention_head_masked_196_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
#                     elif rep == 'context':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_context_attention_head_masked_196_random_run{}/'
#                                 '{}_layer_{}_final_word_context_attention_head_masked_196_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_196_random_run{}/'
#                                 '{}_layer_{}_final_word_literal_attention_head_masked_196_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

#                     rdm = rsa_utils.get_rdm(reps)
#                     # if layer==15:
#                     #     if model_name == 'meta-llama/Llama-3.2-3B':
#                     #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
#                     #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
#                     #         plt.close()
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)

#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# # 3-panel comparison figure: normal vs ablation vs random ablation
# # Each panel overlays standard/context/no_context curves.
# normal_df = pd.read_csv('results/idiom_representations_normal.csv')
# ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
# random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# # Keep plotting consistent with the selected model(s) in this run.
# selected_models = set(model_names)
# normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
# ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
# random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# # create directory for figures if it doesn't exist
# os.makedirs('figures', exist_ok=True)

# label_map = {
#     'standard': 'Neutral',
#     'context': 'Figurative',
#     'no_context': 'Literal',
# }

# condition_palette = {
#     'No Ablation': '#8E44AD',
#     'Idiomaticity Ablation': '#A6761D',
#     'Random Ablation': '#D81B60',
# }

# for df in (normal_df, ablation_df, random_ablation_df):
#     df['Representation'] = df['representation'].map(label_map)
#     # drop any rows with unexpected representation labels
#     df.dropna(subset=['Representation'], inplace=True)

# normal_df['Condition'] = 'No Ablation'
# ablation_df['Condition'] = 'Idiomaticity Ablation'
# random_ablation_df['Condition'] = 'Random Ablation'

# plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

# metric = 'same_relation_group_rdm_corr'
# fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
# fig.suptitle('Qwen Attention Head Ablation 25%, full region')
# panel_order = ['Neutral', 'Figurative', 'Literal']
# region_test = 'ttest'

# for i, rep_name in enumerate(panel_order):
#     ax = axes[i]
#     panel_df = plot_df[plot_df['Representation'] == rep_name]
#     significant_layers = get_significant_layers_for_panel(
#         normal_df,
#         ablation_df,
#         rep_name,
#         metric,
#         test_kind=region_test,
#     )
#     sns.lineplot(
#         data=panel_df,
#         x='layer',
#         y=metric,
#         hue='Condition',
#         style='Condition',
#         markers=True,
#         dashes=False,
#         palette=condition_palette,
#         ax=ax,
#     )
#     add_significance_markers(ax, panel_df, metric, significant_layers)
#     ax.set_title(rep_name)
#     ax.set_xlabel('Layer')
#     if i == 0:
#         ax.set_ylabel('Correlation')
#     else:
#         ax.set_ylabel('')
#     ax.axhline(0, color='black', linestyle='--', linewidth=1)
#     ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

#     configure_panel_legend(ax, i)

# fig.tight_layout(rect=[0, 0, 1, 0.93])
# plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_qwen_196.png', format='png')
# plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_qwen_196.eps', format='eps')
# plt.show()





# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
                       
#                         reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
#                     elif rep == 'context':
                        
#                         reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


#                     rdm = rsa_utils.get_rdm(reps)
                
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)
#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
                       
#                         reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_196/{}_layer_{}_final_word_standard_attention_head_masked_significant_196.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
#                     elif rep == 'context':
                        
#                         reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_196/{}_layer_{}_final_word_context_attention_head_masked_significant_196.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
#                         reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_196/{}_layer_{}_final_word_literal_attention_head_masked_significant_196.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


#                     rdm = rsa_utils.get_rdm(reps)
#                     # if layer==15:
#                     #     if model_name == 'meta-llama/Llama-3.2-3B':
#                     #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
#                     #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
#                     #         plt.close()
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)
#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


# load = False

# if load:
#     relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
#     with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
#         idiom_correlation_dict5 = pickle.load(f)
# else:
#     rows = []
#     i = 0
#     corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

#     representations = ["standard", "context", "no_context"]
#     idiom_correlation_dict5 = {}

#     for model_name in model_names:
#         print(model_name)
#         for layer in tqdm.tqdm(range(1, 33)):

#             if layer > 12 and 'gpt' in model_name:
#                 continue
#             if layer > 12 and 'bert' in model_name:
#                 continue
#             elif layer > 28 and 'llama' in model_name:
#                 continue
#             elif layer > 28 and 'tiiuae' in model_name:
#                 continue
#             elif layer > 28 and 'deepseek' in model_name:
#                 continue
#             elif layer > 28 and 'Qwen2.5' in model_name:
#                 continue


#             if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
#                 for rep in representations:
                
                
                
#                     if rep == 'standard':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_196_random_run{}/'
#                                 '{}_layer_{}_final_word_standard_attention_head_masked_196_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
#                     elif rep == 'context':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_context_attention_head_masked_196_random_run{}/'
#                                 '{}_layer_{}_final_word_context_attention_head_masked_196_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
#                         # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
#                     elif rep == 'no_context':
#                         run_arrays = []
#                         for run_idx in range(1, 6):
#                             path = (
#                                 'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_196_random_run{}/'
#                                 '{}_layer_{}_final_word_literal_attention_head_masked_196_random_run{}.npy'
#                             ).format(
#                                 model_name.split('-')[0],
#                                 layer,
#                                 run_idx,
#                                 model_name,
#                                 layer,
#                                 run_idx,
#                             )
#                             run_arrays.append(np.load(path))
#                         reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

#                     rdm = rsa_utils.get_rdm(reps)
#                     # if layer==15:
#                     #     if model_name == 'meta-llama/Llama-3.2-3B':
#                     #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
#                     #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
#                     #         plt.close()
                  

#                     row = {'model': model_name, 'layer': layer, 'representation': rep}

#                     for target_rdm_name, target_rdm in group_rdms_to_correlate:
#                         second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
#                         res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
#                                                                                           corr_metric=corr_metric, keep_corrs=True)

#                         row = {**row, **res}
#                         #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
#                         if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
#                             idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
#                         rows.append(row)

#     with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
#         pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


#     relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
#     relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# # 3-panel comparison figure: normal vs ablation vs random ablation
# # Each panel overlays standard/context/no_context curves.
# normal_df = pd.read_csv('results/idiom_representations_normal.csv')
# ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
# random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# # Keep plotting consistent with the selected model(s) in this run.
# selected_models = set(model_names)
# normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
# ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
# random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# # create directory for figures if it doesn't exist
# os.makedirs('figures', exist_ok=True)

# label_map = {
#     'standard': 'Neutral',
#     'context': 'Figurative',
#     'no_context': 'Literal',
# }

# condition_palette = {
#     'No Ablation': '#8E44AD',
#     'Idiomaticity Ablation': '#A6761D',
#     'Random Ablation': '#D81B60',
# }

# for df in (normal_df, ablation_df, random_ablation_df):
#     df['Representation'] = df['representation'].map(label_map)
#     # drop any rows with unexpected representation labels
#     df.dropna(subset=['Representation'], inplace=True)

# normal_df['Condition'] = 'No Ablation'
# ablation_df['Condition'] = 'Idiomaticity Ablation'
# random_ablation_df['Condition'] = 'Random Ablation'

# plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

# metric = 'same_relation_group_rdm_corr'
# fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
# fig.suptitle('Qwen Attention Head Ablation 25%, outlined region')
# panel_order = ['Neutral', 'Figurative', 'Literal']
# region_test = 'wilcoxon'

# for i, rep_name in enumerate(panel_order):
#     ax = axes[i]
#     panel_df = plot_df[plot_df['Representation'] == rep_name]
#     significant_layers = get_significant_layers_for_panel(
#         normal_df,
#         ablation_df,
#         rep_name,
#         metric,
#         test_kind=region_test,
#     )
#     sns.lineplot(
#         data=panel_df,
#         x='layer',
#         y=metric,
#         hue='Condition',
#         style='Condition',
#         markers=True,
#         dashes=False,
#         palette=condition_palette,
#         ax=ax,
#     )
#     add_significance_markers(ax, panel_df, metric, significant_layers)
#     ax.set_title(rep_name)
#     ax.set_xlabel('Layer')
#     if i == 0:
#         ax.set_ylabel('Correlation')
#     else:
#         ax.set_ylabel('')
#     ax.axhline(0, color='black', linestyle='--', linewidth=1)
#     ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

#     configure_panel_legend(ax, i)

# fig.tight_layout(rect=[0, 0, 1, 0.93])
# plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_qwen_196_with_mask.png', format='png')
# plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_qwen_196_with_mask.eps', format='eps')
# plt.show()






####


model_names = ["mistralai/Mistral-7B-v0.1"]


load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_102/{}_layer_{}_final_word_standard_attention_head_masked_significant_102.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_102/{}_layer_{}_final_word_context_attention_head_masked_significant_102.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_102/{}_layer_{}_final_word_literal_attention_head_masked_significant_102.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_102_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_102_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_102_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_102_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_102_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_102_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Mistral Attention Head Ablation 10%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_mistral_102.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_mistral_102.eps', format='eps')
plt.show()





load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_102/{}_layer_{}_final_word_standard_attention_head_masked_significant_102.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_102/{}_layer_{}_final_word_context_attention_head_masked_significant_102.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_102/{}_layer_{}_final_word_literal_attention_head_masked_significant_102.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_102_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_102_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_102_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_102_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_102_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_102_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Mistral Attention Head Ablation 10%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_mistral_102_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_mistral_102_with_mask.eps', format='eps')
plt.show()




load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_256/{}_layer_{}_final_word_standard_attention_head_masked_significant_256.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_256/{}_layer_{}_final_word_context_attention_head_masked_significant_256.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_256/{}_layer_{}_final_word_literal_attention_head_masked_significant_256.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_256_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_256_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_256_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_256_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_256_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_256_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Mistral Attention Head Ablation 25%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_mistral_256.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_mistral_256.eps', format='eps')
plt.show()





load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_256/{}_layer_{}_final_word_standard_attention_head_masked_significant_256.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_256/{}_layer_{}_final_word_context_attention_head_masked_significant_256.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_256/{}_layer_{}_final_word_literal_attention_head_masked_significant_256.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_256_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_256_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_256_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_256_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_256_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_256_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Mistral Attention Head Ablation 25%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_mistral_256_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_mistral_256_with_mask.eps', format='eps')
plt.show()



#####


model_names = ["tiiuae/Falcon3-7B-Base"]


load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_67/{}_layer_{}_final_word_standard_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_67/{}_layer_{}_final_word_context_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_67/{}_layer_{}_final_word_literal_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Falcon Attention Head Ablation 10%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_falcon_67.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_falcon_67.eps', format='eps')
plt.show()





load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_67/{}_layer_{}_final_word_standard_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_67/{}_layer_{}_final_word_context_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_67/{}_layer_{}_final_word_literal_attention_head_masked_significant_67.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_67_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_67_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Falcon Attention Head Ablation 10%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_falcon_67_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_falcon_67_with_mask.eps', format='eps')
plt.show()




load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_168/{}_layer_{}_final_word_standard_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_168/{}_layer_{}_final_word_context_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_168/{}_layer_{}_final_word_literal_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Falcon Attention Head Ablation 25%, full region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'ttest'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_falcon_168.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_falcon_168.eps', format='eps')
plt.show()





load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_normal.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard/{}_layer_{}_final_word_standard.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context/{}_layer_{}_final_word_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context/{}_layer_{}_final_word_no_context.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_normal.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_attention_head_masked_significant_168/{}_layer_{}_final_word_standard_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'context':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_attention_head_masked_significant_168/{}_layer_{}_final_word_context_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_literal_attention_head_masked_significant_168/{}_layer_{}_final_word_literal_attention_head_masked_significant_168.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation.csv')


load = False

if load:
    relation_results_within_compound_groups_per_word_df_ablation = pd.read_csv('results/idiom_representations_ablation_random.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "context", "no_context"]
    idiom_correlation_dict5 = {}

    for model_name in model_names:
        print(model_name)
        for layer in tqdm.tqdm(range(1, 33)):

            if layer > 12 and 'gpt' in model_name:
                continue
            if layer > 12 and 'bert' in model_name:
                continue
            elif layer > 28 and 'llama' in model_name:
                continue
            elif layer > 28 and 'tiiuae' in model_name:
                continue
            elif layer > 28 and 'deepseek' in model_name:
                continue
            elif layer > 28 and 'Qwen2.5' in model_name:
                continue


            if model_name in ['mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 'openai-community/gpt2', "Qwen/Qwen2.5-7B"]:
                
            
                for rep in representations:
                
                
                
                    if rep == 'standard':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_standard_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_standard_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                     
                    elif rep == 'context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_context_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_context_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)
                       
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'no_context':
                        run_arrays = []
                        for run_idx in range(1, 6):
                            path = (
                                'data/representations/{}/layer_{}/final_word_literal_attention_head_masked_168_random_run{}/'
                                '{}_layer_{}_final_word_literal_attention_head_masked_168_random_run{}.npy'
                            ).format(
                                model_name.split('-')[0],
                                layer,
                                run_idx,
                                model_name,
                                layer,
                                run_idx,
                            )
                            run_arrays.append(np.load(path))
                        reps = np.mean(np.stack(run_arrays, axis=0), axis=0)

                    rdm = rsa_utils.get_rdm(reps)
                    # if layer==15:
                    #     if model_name == 'meta-llama/Llama-3.2-3B':
                    #         rsa_utils.plot_mtx(rdm[:15, :15], '{} {} {}'.format(model_name, layer, rep))
                    #         plt.savefig('figures/rdm_meta_{}_{}.png'.format(layer, rep), format='png')
                    #         plt.close()
                  

                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'
                       
                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'standard':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_ablation_random.csv')



# 3-panel comparison figure: normal vs ablation vs random ablation
# Each panel overlays standard/context/no_context curves.
normal_df = pd.read_csv('results/idiom_representations_normal.csv')
ablation_df = pd.read_csv('results/idiom_representations_ablation.csv')
random_ablation_df = pd.read_csv('results/idiom_representations_ablation_random.csv')

# Keep plotting consistent with the selected model(s) in this run.
selected_models = set(model_names)
normal_df = normal_df[normal_df['model'].isin(selected_models)].copy()
ablation_df = ablation_df[ablation_df['model'].isin(selected_models)].copy()
random_ablation_df = random_ablation_df[random_ablation_df['model'].isin(selected_models)].copy()

# create directory for figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

label_map = {
    'standard': 'Neutral',
    'context': 'Figurative',
    'no_context': 'Literal',
}

condition_palette = {
    'No Ablation': '#8E44AD',
    'Idiomaticity Ablation': '#A6761D',
    'Random Ablation': '#D81B60',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'No Ablation'
ablation_df['Condition'] = 'Idiomaticity Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle('Falcon Attention Head Ablation 25%, outlined region')
panel_order = ['Neutral', 'Figurative', 'Literal']
region_test = 'wilcoxon'

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
    significant_layers = get_significant_layers_for_panel(
        normal_df,
        ablation_df,
        rep_name,
        metric,
        test_kind=region_test,
    )
    sns.lineplot(
        data=panel_df,
        x='layer',
        y=metric,
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=False,
        palette=condition_palette,
        ax=ax,
    )
    add_significance_markers(ax, panel_df, metric, significant_layers)
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    configure_panel_legend(ax, i)

fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_falcon_168_with_mask.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_attention_head_falcon_168_with_mask.eps', format='eps')
plt.show()












