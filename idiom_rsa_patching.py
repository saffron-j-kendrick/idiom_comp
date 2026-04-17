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
from itertools import product
import pandas as pd
import xlsxwriter
from matplotlib.lines import Line2D
import rsa_utils
import data_utils



## DATA

df = pd.read_excel("data/standard_sentences.xlsx")
corr_metric = 'kendalltau'
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
    'standard': 'Standard',
    'context': 'Context',
    'no_context': 'No Context',
}

condition_palette = {
    'Normal': '#1E88E5',
    'Ablation': '#43A047',
    'Random Ablation': '#F4511E',
}

for df in (normal_df, ablation_df, random_ablation_df):
    df['Representation'] = df['representation'].map(label_map)
    # drop any rows with unexpected representation labels
    df.dropna(subset=['Representation'], inplace=True)

normal_df['Condition'] = 'Normal'
ablation_df['Condition'] = 'Ablation'
random_ablation_df['Condition'] = 'Random Ablation'

plot_df = pd.concat([normal_df, ablation_df, random_ablation_df], ignore_index=True)

metric = 'same_relation_group_rdm_corr'
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
panel_order = ['Standard', 'Context', 'No Context']

for i, rep_name in enumerate(panel_order):
    ax = axes[i]
    panel_df = plot_df[plot_df['Representation'] == rep_name]
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
    ax.set_title(rep_name)
    ax.set_xlabel('Layer')
    if i == 0:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel('')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x for x in sorted(panel_df['layer'].unique()) if x % 4 == 0])

    legend = ax.get_legend()
    if i == 2 and legend is not None:
        legend.set_title('')
        legend.set_bbox_to_anchor((1.02, 1))
        legend._loc = 2  # upper left
    elif legend is not None:
        legend.remove()

fig.tight_layout()
plt.savefig('figures/idioms_ablation_three_panel_comparison_67_mlp_masked_67.png', format='png')
plt.savefig('figures/idioms_ablation_three_panel_comparison_67_mlp_masked_67.eps', format='eps')
plt.show()



