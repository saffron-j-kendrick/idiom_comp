
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
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import pandas as pd
import xlsxwriter
import os
from matplotlib.lines import Line2D
import rsa_utils
import data_utils



## DATA

df = pd.read_excel("data/standard_sentences.xlsx")
corr_metric = 'kendalltau'
order_dict = dict(zip(["openai-community/gpt2", 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B","mistralai/Mistral-7B-v0.1"], range(0, 6)))
model_name_map = {'meta-llama/Llama-3.2-3B' : 'Llama-3.2-3B', "openai-community/gpt2" : "GPT2", "tiiuae/Falcon3-7B-Base" : "Falcon3-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" : "DeepSeek-R1-1.5B", "Qwen/Qwen2.5-7B" : "Qwen2.5-7B", "mistralai/Mistral-7B-v0.1" : "Mistral-7B"}



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




# model_names = ["openai-community/gpt2", 'meta-llama/Llama-3.2-3B', "tiiuae/Falcon3-7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B","mistralai/Mistral-7B-v0.1"]

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

# labels_within_group = np.array([
#     [1, 1, 1, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 1, 1],
# ])

same_relation_group_rdm = rsa_utils.get_rdm(labels_within_group)
rsa_utils.plot_mtx(same_relation_group_rdm, "")

current_cmap = plt.get_cmap('Spectral_r')
current_cmap.set_bad(color='gray')

start = 1
sent_inds = np.arange(start * 8, start * 8 + 8)

fig, ax = plt.subplots(figsize=(8, 6))
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
# plt.savefig('figures/patching_relation_group_rdm.png', bbox_inches='tight')
# plt.savefig('figures/patching_relation_group_rdm.eps', format='eps', bbox_inches='tight')
plt.savefig('figures/same_literal_meaning_ground_truth_rdm.png', bbox_inches='tight')
plt.savefig('figures/same_literal_meaning_ground_truth_rdm.eps', format='eps', bbox_inches='tight')


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

model_names = ["openai-community/gpt2", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "meta-llama/Llama-3.2-3B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base","Qwen/Qwen2.5-7B"]




load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_standard.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["final_verb_token", "final_noun_token", "final_word_token"]
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
                
                
                
                    if rep == 'final_word_token':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_v2/{}_layer_{}_final_word_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'final_noun_token':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_head_standard_v2/{}_layer_{}_final_head_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'final_verb_token':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_modifier_standard_v2/{}_layer_{}_final_modifier_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


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
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word_token':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_standard.csv')




standard_representation_colors = {
    "final_verb_token": "#C0392B",  # Red
    "final_noun_token": "#8E44AD",  # Between red and blue
    "final_word_token": "#1F77B4",  # Blue (distinct from context blue)
}
standard_representation_order = ["final_verb_token", "final_noun_token", "final_word_token"]

for y, ylim in zip(['same_relation_group_rdm_corr'], [(-0.4, 1)]):
    #fig_shape = (2, 3)
    fig_shape = (2, 3)
    fig_df = relation_results_within_compound_groups_per_word_df
    fig_df = sort_df_by_model_order(fig_df)
    
    print(y)
    single_fig_size = 4
    fig = plt.figure(figsize=(fig_shape[1] * 3.25, fig_shape[0] * 3.25, ))
    fig.suptitle('Idiom RSA with neutral dataset, full upper triangular region', fontsize=12)


    grid_height = fig_shape[0] * single_fig_size
    grid_width = fig_shape[1] * single_fig_size
    grid = fig.add_gridspec(grid_height, grid_width, hspace=5, wspace=3)

    for (i, model_df), (fig_y, fig_x) in zip(fig_df.groupby('model_order'), product(range(fig_shape[0]), range(fig_shape[1]))):
        #model_df = sort_df_by_model_order(model_df)
        # print(model, fig_coords)
        model = model_df.iloc[0].model
        line_plot_ax = fig.add_subplot(grid[fig_y*single_fig_size:fig_y*single_fig_size+single_fig_size, fig_x*single_fig_size:fig_x*single_fig_size+single_fig_size]);
        g = sns.lineplot(
            data=model_df,
            hue='representation',
            y=y,
            x='layer',
            style='representation',
            hue_order=standard_representation_order,
            style_order=standard_representation_order,
            markers=True,
            palette=standard_representation_colors,
            ax=line_plot_ax
        );
        
        line_plot_ax.set_xticks([layer for layer in model_df.layer.unique() if layer % 4 == 0])


        #line_plot_ax.set_xticks(model_df.layer.unique());
#         g.set(ylim=)
        g.set(ylim=ylim);

        if fig_x == 0:
            g.set_ylabel('Correlation');
        else:
            g.set_ylabel('');
        
            
        if fig_y == 0:
            g.set_xlabel('');
        if fig_x == 0 and fig_y == 0:
            g.legend(loc='best')
        else:
            legend = g.get_legend()
            if legend is not None:
                legend.remove()
            
        g.axhline(0, color='black', linestyle='--', linewidth=1)

        g.set_title(model, fontsize = 9);
    plt.tight_layout()
    plt.savefig('figures/idioms_standard_v2.png', format = 'png')
    plt.savefig('figures/idioms_standard_v2.eps', format='eps')
    fig.show()
    
plt.tight_layout()
plt.show()






load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_standard.csv')
    with open('idiom_correlation_dict_standard.pkl', 'rb') as f:
        idiom_correlation_dict5 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["final_verb_token", "final_noun_token", "final_word_token"]
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
                
                
                
                    if rep == 'final_word_token':
                       
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_v2/{}_layer_{}_final_word_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                     
                    elif rep == 'final_noun_token':
                        
                        reps = np.load('data/representations/{}/layer_{}/final_head_standard_v2/{}_layer_{}_final_head_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_head/{}_layer_{}_final
                    elif rep == 'final_verb_token':
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_modifier/{}_layer_{}_final_modifier_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_modifier_standard_v2/{}_layer_{}_final_modifier_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                      


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
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word_token':
                            idiom_correlation_dict5['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_standard.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict5, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_standard.csv')


standard_representation_colors = {
    "final_verb_token": "#C0392B",  # Red
    "final_noun_token": "#8E44AD",  # Between red and blue
    "final_word_token": "#1F77B4",  # Blue (distinct from context blue)
}
standard_representation_order = ["final_verb_token", "final_noun_token", "final_word_token"]

for y, ylim in zip(['same_relation_group_rdm_corr'], [(-0.4, 1)]):
    #fig_shape = (2, 3)
    fig_shape = (2, 3)
    fig_df = relation_results_within_compound_groups_per_word_df
    fig_df = sort_df_by_model_order(fig_df)
    
    print(y)
    single_fig_size = 4
    fig = plt.figure(figsize=(fig_shape[1] * 3.25, fig_shape[0] * 3.25, ))
    fig.suptitle('Idiom RSA with neutral dataset, full upper triangular region', fontsize=12)


    grid_height = fig_shape[0] * single_fig_size
    grid_width = fig_shape[1] * single_fig_size
    grid = fig.add_gridspec(grid_height, grid_width, hspace=5, wspace=3)

    for (i, model_df), (fig_y, fig_x) in zip(fig_df.groupby('model_order'), product(range(fig_shape[0]), range(fig_shape[1]))):
        #model_df = sort_df_by_model_order(model_df)
        # print(model, fig_coords)
        model = model_df.iloc[0].model
        line_plot_ax = fig.add_subplot(grid[fig_y*single_fig_size:fig_y*single_fig_size+single_fig_size, fig_x*single_fig_size:fig_x*single_fig_size+single_fig_size]);
        g = sns.lineplot(
            data=model_df,
            hue='representation',
            y=y,
            x='layer',
            style='representation',
            hue_order=standard_representation_order,
            style_order=standard_representation_order,
            markers=True,
            palette=standard_representation_colors,
            ax=line_plot_ax
        );
        
        line_plot_ax.set_xticks([layer for layer in model_df.layer.unique() if layer % 4 == 0])


        #line_plot_ax.set_xticks(model_df.layer.unique());
#         g.set(ylim=)
        g.set(ylim=ylim);

        if fig_x == 0:
            g.set_ylabel('Correlation');
        else:
            g.set_ylabel('');
        
            
        if fig_y == 0:
            g.set_xlabel('');
        if fig_x == 0 and fig_y == 0:
            g.legend(loc='best')
        else:
            legend = g.get_legend()
            if legend is not None:
                legend.remove()
            
        g.axhline(0, color='black', linestyle='--', linewidth=1)

        g.set_title(model, fontsize = 9);
    plt.tight_layout()
    plt.savefig('figures/idioms_standard_v2_with_mask.png', format = 'png')
    plt.savefig('figures/idioms_standard_v2_with_mask.eps', format='eps')
    fig.show()
    
plt.tight_layout()
plt.show()






### 




load = False

if load:
    relation_results_within_compound_groups_per_word_df1 = pd.read_csv('results/idiom_representations_final_words1.csv')
    with open('idiom_correlation_dict_final_word1.pkl', 'rb') as f:
        idiom_correlation_dict1 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["final_word"]
    idiom_correlation_dict1 = {}

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
                
                
                
                    if rep == 'final_word':
                    
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_word_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_v2/{}_layer_{}_final_word_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_but_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_that = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_that_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
            
                        # reps = np.mean([reps_and, reps_but, reps_that], axis=0)

                    rdm = rsa_utils.get_rdm(reps)


                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'

                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word':
                            idiom_correlation_dict1['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_final_word1.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict1, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df1 = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df1.to_csv('results/idiom_representations_final_words1.csv')



load = False

if load:
    relation_results_within_compound_groups_per_word_df2 = pd.read_csv('results/idiom_representations_final_words2.csv')
    with open('idiom_correlation_dict_final_word2.pkl', 'rb') as f:
        idiom_correlation_dict2 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["final_word"]
    idiom_correlation_dict2 = {}

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
                
                
                
                    if rep == 'final_word':
                    
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_word_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context_v2/{}_layer_{}_final_word_no_context_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_but_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_that = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_that_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
            
                        # reps = np.mean([reps_and, reps_but, reps_that], axis=0)

                    rdm = rsa_utils.get_rdm(reps)


                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'

                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word':
                            idiom_correlation_dict2['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_final_word2.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict2, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df2 = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df2.to_csv('results/idiom_representations_final_words2.csv')




load = False

if load:
    relation_results_within_compound_groups_per_word_df3 = pd.read_csv('results/idiom_representations_final_words3.csv')
    with open('idiom_correlation_dict_final_word3.pkl', 'rb') as f:
        idiom_correlation_dict3 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["final_word"]
    idiom_correlation_dict3 = {}

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
                
                
                
                    if rep == 'final_word':
                    
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_word_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_v2/{}_layer_{}_final_word_context_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_but_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_that = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_that_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
            
                        # reps = np.mean([reps_and, reps_but, reps_that], axis=0)

                    rdm = rsa_utils.get_rdm(reps)


                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'

                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word':
                            idiom_correlation_dict3['{}_{}'.format(model_name, layer)] = corrs  
                        
                        rows.append(row)

    with open('idiom_correlation_dict_final_word3.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict3, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df3 = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df3.to_csv('results/idiom_representations_final_words3.csv')







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
    fdr_method = 'negcorr' # 'indep' or 'negcorr
    df.loc[df.p > -1, 'corrected_p'] = fdrcorrection(df[df.p > -1].p.tolist(), method=fdr_method, alpha=fdr_error_rate)[-1]
    df.loc[df.p == -1, 'corrected_p'] = -1
    df['formatted_corrected_p']  = list(map(format_p_val, df.corrected_p))
    
    return df

rows = []

for model in model_names:
    for layer in list(range(1, 33)):
        row = {'model': model, 'layer': layer}
        
        if 'gpt' in model and layer > 12:
            row['p'] = -1
        elif 'llama' in model and layer > 28:
            row['p'] = -1
        elif 'tiiuae' in model and layer > 28:
            row['p'] = -1
        elif 'deepseek' in model and layer > 28:
            row['p'] = -1
        elif 'Qwen2.5' in model and layer > 28:
            row['p'] = -1
        
        else:


            group_correlations_processed_context = idiom_correlation_dict3['{}_{}'.format(model, layer)]
            group_correlations_processed_no_context = idiom_correlation_dict2['{}_{}'.format(model, layer)]
            group_correlations_processed_standard = idiom_correlation_dict1['{}_{}'.format(model, layer)]
            row['p'] =  ttest_rel(group_correlations_processed_context, group_correlations_processed_standard, alternative='greater').pvalue
            
        rows.append(row)
        
paired_t_test_df = pd.DataFrame(rows)

paired_t_test_df = fdr_correction(paired_t_test_df)

# Create a second dataframe for the Figurative vs No-Context comparison
rows_context_vs_no = []

for model in model_names:
    for layer in list(range(1, 33)):
        row = {'model': model, 'layer': layer}
        
        # Keep same layer constraints as your previous block
        if (('gpt' in model and layer > 12) or 
            (any(m in model for m in ['llama', 'tiiuae', 'deepseek', 'Qwen2.5']) and layer > 28)):
            row['p'] = -1
        else:
            # Comparison: Figurative vs Literal/No-Context
            fig_corrs = idiom_correlation_dict3[f'{model}_{layer}']
            no_context_corrs = idiom_correlation_dict2[f'{model}_{layer}']
            
            row['p'] = ttest_rel(fig_corrs, no_context_corrs, alternative='greater').pvalue
            
        rows_context_vs_no.append(row)
        
paired_t_test_context_vs_no_df = pd.DataFrame(rows_context_vs_no)
paired_t_test_context_vs_no_df = fdr_correction(paired_t_test_context_vs_no_df)


load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_final_words_context_comparison.csv')
    with open('idiom_correlation_dict_final_word.pkl', 'rb') as f:
        idiom_correlation_dict = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "figurative_context", "literal_context"]
    idiom_correlation_dict = {}

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
                    
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_word_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_v2/{}_layer_{}_final_word_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_but_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_that = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_that_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
            
                        # reps = np.mean([reps_and, reps_but, reps_that], axis=0)


                    elif rep == "literal_context":

                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context_v2/{}_layer_{}_final_word_no_context_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))

                    elif rep == "figurative_context":
                    
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_v2/{}_layer_{}_final_word_context_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                    
                    # elif rep == "preceding_word":
                    
                    #     reps = np.load('data/representations/{}/layer_{}/b_word_single_literal/{}_layer_{}_b_word_single_literal.npy'.format(model_name.split('-')[0], layer, model_name, layer))



                    rdm = rsa_utils.get_rdm(reps)

                    
                       


                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'

                        res, corrs = rsa_utils.correlation_and_rows(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word':
                            idiom_correlation_dict['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_final_word.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_final_words_context_comparison.csv')







representation_colors = {
    "Figurative Context": "#3B5BDB",  # Indigo-blue
    "Neutral Context": "#6C757D",  # Gray
    "Literal Context": "#E03131"  # Red
}

for y, ylim in zip(['same_relation_group_rdm_corr'], [(-0.4, 1.00)]):
    #fig_shape = (2, 3)
    fig_shape = (2, 3)

    # Keep only final-word rows and support both legacy/new naming.
    final_word_labels = ['final_word_token', 'final_word']
    rows = relation_results_within_compound_groups_per_word_df3[
        relation_results_within_compound_groups_per_word_df3.representation.isin(final_word_labels)
    ].copy()
    neutral_rows = relation_results_within_compound_groups_per_word_df1[
        relation_results_within_compound_groups_per_word_df1.representation.isin(final_word_labels)
    ].copy()
    literal_rows = relation_results_within_compound_groups_per_word_df2[
        relation_results_within_compound_groups_per_word_df2.representation.isin(final_word_labels)
    ].copy()

    rows['Processing condition'] = 'Figurative Context'
    neutral_rows['Processing condition'] = 'Neutral Context'
    literal_rows['Processing condition'] = 'Literal Context'

    fig_df = pd.concat([rows, neutral_rows, literal_rows])   
    fig_df = sort_df_by_model_order(fig_df)
   


    
    single_fig_size = 4
    fig = plt.figure(figsize=(fig_shape[1] * 3.25, fig_shape[0] * 3.25, ))
    fig.suptitle('Idiom RSA with figurative, neutral, and literal context dataset, full upper triangular region', fontsize=12)

    grid_height = fig_shape[0] * single_fig_size
    grid_width = fig_shape[1] * single_fig_size
    grid = fig.add_gridspec(grid_height, grid_width, hspace=5, wspace=3)
    sig_legend_handles = [
        Line2D([0], [0], marker='*', linestyle='None', markersize=10, color='#6C757D',
               label='p<=0.05: Figurative vs Neutral Context'),
        Line2D([0], [0], marker='*', linestyle='None', markersize=10, color='#E03131',
               label='p<=0.05: Figurative vs Literal Context'),
    ]
    

    for (i, model_df), (fig_y, fig_x) in zip(fig_df.groupby('model_order'), product(range(fig_shape[0]), range(fig_shape[1]))):
        model_df = sort_df_by_model_order(model_df)
#         print(model, fig_coords)
        model = model_df.iloc[0].model
        model_name = model_df.iloc[0].model_name
    
        line_plot_ax = fig.add_subplot(grid[fig_y*single_fig_size:fig_y*single_fig_size+single_fig_size, fig_x*single_fig_size:fig_x*single_fig_size+single_fig_size]);
        g = sns.lineplot(data=model_df, hue='Processing condition', y=y, x='layer', style='Processing condition', markers=True, palette= representation_colors, ax=line_plot_ax);
        # line_plot_ax.set_xticks(model_df.layer.unique());
        line_plot_ax.set_xticks([layer for layer in model_df.layer.unique() if layer % 4 == 0])


#         g.set(ylim=)
        g.set(ylim=ylim);

        if fig_x == 0:
            g.set_ylabel('Correlation');
        else:
            g.set_ylabel('');
            
        if fig_y == 0:
            g.set_xlabel('');

        # if fig_x == 0 and fig_y == 0:
        #     g.legend(loc='best', bbox_to_anchor=(1.5, -2, 0.5, 0.5), ncol=1);
        # else:
        #     g.legend().remove();
        existing_handles, existing_labels = g.get_legend_handles_labels()
        legend = g.get_legend()
        if fig_x == 0 and fig_y == 0:
            g.legend(
                existing_handles + sig_legend_handles,
                existing_labels + [h.get_label() for h in sig_legend_handles],
                loc='upper right',
                prop={'size': 7},
                markerscale=0.75
            )
        elif legend is not None:
            legend.remove()

        
        g.axhline(0, color='black', linestyle='--', linewidth=1)
        
        g.set_title(model_name);
        
        # for layer in model_df.layer.unique():
        #     pval = paired_t_test_df[(paired_t_test_df.model == model) & (paired_t_test_df.layer == layer)]['corrected_p'].iloc[0]
        #     x_offset =  0.075 if 'distil' in model else 0.2
        #     y_offset = 0.1 - 0.035

        #     if pval <= 0.05:
        #         y_val = model_df[(model_df.layer==layer)]['same_relation_group_rdm_corr'].min()
        #         plt.text(layer - x_offset, 0 - y_offset, '*', weight='bold', size='x-large', color='black')
        for layer in model_df.layer.unique():
            # Existing Comparison (Figurative vs Standard)
            pval1 = paired_t_test_df[(paired_t_test_df.model == model) & (paired_t_test_df.layer == layer)]['corrected_p'].iloc[0]
            
            # New Comparison (Figurative vs No Context)
            pval2 = paired_t_test_context_vs_no_df[(paired_t_test_context_vs_no_df.model == model) & (paired_t_test_context_vs_no_df.layer == layer)]['corrected_p'].iloc[0]
            
            x_offset = 0.2
            y_base = -0.05 # Adjust based on your ylim
            
            # Draw Asterisk 1: Figurative vs Standard (Green color to match 'Standard' line?)
            if 0 <= pval1 <= 0.05:
                plt.text(layer - x_offset, y_base, '*', weight='bold', size='large', color='#6C757D')
            
            # Draw Asterisk 2: Figurative vs No Context (Red color to match 'Literal' line?)
            # Positioned slightly lower
            if 0 <= pval2 <= 0.05:
                plt.text(layer - x_offset, y_base - 0.05, '*', weight='bold', size='large', color='#E03131')
        
    fig.tight_layout()
    fig.show()

plt.savefig('figures/idioms_context_comparison_v2.png', format = 'png')
plt.savefig('figures/idioms_context_comparison_v2.eps', format='eps')





#

load = False

if load:
    relation_results_within_compound_groups_per_word_df1 = pd.read_csv('results/idiom_representations_final_words1.csv')
    with open('idiom_correlation_dict_final_word1.pkl', 'rb') as f:
        idiom_correlation_dict1 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["final_word"]
    idiom_correlation_dict1 = {}

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
                
                
                
                    if rep == 'final_word':
                    
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_word_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_v2/{}_layer_{}_final_word_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_but_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_that = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_that_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
            
                        # reps = np.mean([reps_and, reps_but, reps_that], axis=0)

                    rdm = rsa_utils.get_rdm(reps)


                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'

                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word':
                            idiom_correlation_dict1['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_final_word1.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict1, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df1 = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df1.to_csv('results/idiom_representations_final_words1.csv')



load = False

if load:
    relation_results_within_compound_groups_per_word_df2 = pd.read_csv('results/idiom_representations_final_words2.csv')
    with open('idiom_correlation_dict_final_word2.pkl', 'rb') as f:
        idiom_correlation_dict2 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["final_word"]
    idiom_correlation_dict2 = {}

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
                
                
                
                    if rep == 'final_word':
                    
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_word_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context_v2/{}_layer_{}_final_word_no_context_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_but_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_that = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_that_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
            
                        # reps = np.mean([reps_and, reps_but, reps_that], axis=0)

                    rdm = rsa_utils.get_rdm(reps)


                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'

                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)

                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word':
                            idiom_correlation_dict2['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_final_word2.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict2, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df2 = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df2.to_csv('results/idiom_representations_final_words2.csv')




load = False

if load:
    relation_results_within_compound_groups_per_word_df3 = pd.read_csv('results/idiom_representations_final_words3.csv')
    with open('idiom_correlation_dict_final_word3.pkl', 'rb') as f:
        idiom_correlation_dict3 = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["final_word"]
    idiom_correlation_dict3 = {}

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
                
                
                
                    if rep == 'final_word':
                    
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_word_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_v2/{}_layer_{}_final_word_context_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_but_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_that = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_that_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
            
                        # reps = np.mean([reps_and, reps_but, reps_that], axis=0)

                    rdm = rsa_utils.get_rdm(reps)


                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'

                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word':
                            idiom_correlation_dict3['{}_{}'.format(model_name, layer)] = corrs  
                        
                        rows.append(row)

    with open('idiom_correlation_dict_final_word3.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict3, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df3 = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df3.to_csv('results/idiom_representations_final_words3.csv')







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
    fdr_method = 'negcorr' # 'indep' or 'negcorr
    df.loc[df.p > -1, 'corrected_p'] = fdrcorrection(df[df.p > -1].p.tolist(), method=fdr_method, alpha=fdr_error_rate)[-1]
    df.loc[df.p == -1, 'corrected_p'] = -1
    df['formatted_corrected_p']  = list(map(format_p_val, df.corrected_p))
    
    return df

rows = []

for model in model_names:
    for layer in list(range(1, 33)):
        row = {'model': model, 'layer': layer}
        
        if 'gpt' in model and layer > 12:
            row['p'] = -1
        elif 'llama' in model and layer > 28:
            row['p'] = -1
        elif 'tiiuae' in model and layer > 28:
            row['p'] = -1
        elif 'deepseek' in model and layer > 28:
            row['p'] = -1
        elif 'Qwen2.5' in model and layer > 28:
            row['p'] = -1
        
        else:


            group_correlations_processed_context = idiom_correlation_dict3['{}_{}'.format(model, layer)]
            group_correlations_processed_no_context = idiom_correlation_dict2['{}_{}'.format(model, layer)]
            group_correlations_processed_standard = idiom_correlation_dict1['{}_{}'.format(model, layer)]
            row['p'] =  ttest_rel(group_correlations_processed_context, group_correlations_processed_standard, alternative='greater').pvalue
            
        rows.append(row)
        
paired_t_test_df = pd.DataFrame(rows)

paired_t_test_df = fdr_correction(paired_t_test_df)

# Create a second dataframe for the Figurative vs No-Context comparison
rows_context_vs_no = []

for model in model_names:
    for layer in list(range(1, 33)):
        row = {'model': model, 'layer': layer}
        
        # Keep same layer constraints as your previous block
        if (('gpt' in model and layer > 12) or 
            (any(m in model for m in ['llama', 'tiiuae', 'deepseek', 'Qwen2.5']) and layer > 28)):
            row['p'] = -1
        else:
            # Comparison: Figurative vs Literal/No-Context
            fig_corrs = idiom_correlation_dict3[f'{model}_{layer}']
            no_context_corrs = idiom_correlation_dict2[f'{model}_{layer}']
            
            row['p'] = ttest_rel(fig_corrs, no_context_corrs, alternative='greater').pvalue
            
        rows_context_vs_no.append(row)
        
paired_t_test_context_vs_no_df = pd.DataFrame(rows_context_vs_no)
paired_t_test_context_vs_no_df = fdr_correction(paired_t_test_context_vs_no_df)


load = False

if load:
    relation_results_within_compound_groups_per_word_df = pd.read_csv('results/idiom_representations_final_words_context_comparison.csv')
    with open('idiom_correlation_dict_final_word.pkl', 'rb') as f:
        idiom_correlation_dict = pickle.load(f)
else:
    rows = []
    i = 0
    corr = lambda x,y: rsa_utils.correlate_rdms(x, y, correlation=corr_metric)

    representations = ["standard", "figurative_context", "literal_context"]
    idiom_correlation_dict = {}

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
                    
                        #reps = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_word_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        reps = np.load('data/representations/{}/layer_{}/final_word_standard_v2/{}_layer_{}_final_word_standard_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_but = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_but_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                        # reps_that = np.load('/Volumes/My Passport/NOUN-NOUN-COMPOUNDS-V1/data/representations/{}/layer_{}/final_word/{}_layer_{}_final_that_tokens.npy'.format(model_name.split('-')[0], layer, model_name, layer))
            
                        # reps = np.mean([reps_and, reps_but, reps_that], axis=0)


                    elif rep == "literal_context":

                        reps = np.load('data/representations/{}/layer_{}/final_word_no_context_v2/{}_layer_{}_final_word_no_context_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))

                    elif rep == "figurative_context":
                    
                        reps = np.load('data/representations/{}/layer_{}/final_word_context_v2/{}_layer_{}_final_word_context_v2.npy'.format(model_name.split('-')[0], layer, model_name, layer))
                    
                    # elif rep == "preceding_word":
                    
                    #     reps = np.load('data/representations/{}/layer_{}/b_word_single_literal/{}_layer_{}_b_word_single_literal.npy'.format(model_name.split('-')[0], layer, model_name, layer))



                    rdm = rsa_utils.get_rdm(reps)

                    
                       


                    row = {'model': model_name, 'layer': layer, 'representation': rep}

                    for target_rdm_name, target_rdm in group_rdms_to_correlate:
                        second_rdm_group_level_already = target_rdm_name == 'same_relation_group_rdm'

                        res, corrs = rsa_utils.correlation_outlined_only(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already,
                                                                                          corr_metric=corr_metric, keep_corrs=True)
                        row = {**row, **res}
                        #row = {**row, **rsa_utils.correlate_over_groups_and_get_row_values(rdm, target_rdm, target_rdm_name, second_rdm_group_level_already=second_rdm_group_level_already, corr_metric=corr_metric)}
                        
                        if target_rdm_name == 'same_relation_group_rdm' and rep == 'final_word':
                            idiom_correlation_dict['{}_{}'.format(model_name, layer)] = corrs 
                        
                        rows.append(row)

    with open('idiom_correlation_dict_final_word.pkl', 'wb') as f:
        pickle.dump(idiom_correlation_dict, f, pickle.HIGHEST_PROTOCOL)


    relation_results_within_compound_groups_per_word_df = pd.DataFrame(rows)
    relation_results_within_compound_groups_per_word_df.to_csv('results/idiom_representations_final_words_context_comparison.csv')








representation_colors = {
    "Figurative Context": "#3B5BDB",  # Indigo-blue
    "Neutral Context": "#6C757D",  # Gray
    "Literal Context": "#E03131"  # Red
}

for y, ylim in zip(['same_relation_group_rdm_corr'], [(-0.4, 1.00)]):
    #fig_shape = (2, 3)
    fig_shape = (2, 3)

    # Keep only final-word rows and support both legacy/new naming.
    final_word_labels = ['final_word_token', 'final_word']
    rows = relation_results_within_compound_groups_per_word_df3[
        relation_results_within_compound_groups_per_word_df3.representation.isin(final_word_labels)
    ].copy()
    neutral_rows = relation_results_within_compound_groups_per_word_df1[
        relation_results_within_compound_groups_per_word_df1.representation.isin(final_word_labels)
    ].copy()
    literal_rows = relation_results_within_compound_groups_per_word_df2[
        relation_results_within_compound_groups_per_word_df2.representation.isin(final_word_labels)
    ].copy()

    rows['Processing condition'] = 'Figurative Context'
    neutral_rows['Processing condition'] = 'Neutral Context'
    literal_rows['Processing condition'] = 'Literal Context'

    fig_df = pd.concat([rows, neutral_rows, literal_rows])   
    fig_df = sort_df_by_model_order(fig_df)
   


    
    single_fig_size = 4
    fig = plt.figure(figsize=(fig_shape[1] * 3.25, fig_shape[0] * 3.25, ))
    fig.suptitle('Idiom RSA with figurative, neutral, and literal context dataset, outlined region', fontsize=12)

    grid_height = fig_shape[0] * single_fig_size
    grid_width = fig_shape[1] * single_fig_size
    grid = fig.add_gridspec(grid_height, grid_width, hspace=5, wspace=3)
    sig_legend_handles = [
        Line2D([0], [0], marker='*', linestyle='None', markersize=10, color='#6C757D',
               label='p<=0.05: Figurative vs Neutral Context'),
        Line2D([0], [0], marker='*', linestyle='None', markersize=10, color='#E03131',
               label='p<=0.05: Figurative vs Literal Context'),
    ]
    

    for (i, model_df), (fig_y, fig_x) in zip(fig_df.groupby('model_order'), product(range(fig_shape[0]), range(fig_shape[1]))):
        model_df = sort_df_by_model_order(model_df)
#         print(model, fig_coords)
        model = model_df.iloc[0].model
        model_name = model_df.iloc[0].model_name
    
        line_plot_ax = fig.add_subplot(grid[fig_y*single_fig_size:fig_y*single_fig_size+single_fig_size, fig_x*single_fig_size:fig_x*single_fig_size+single_fig_size]);
        g = sns.lineplot(data=model_df, hue='Processing condition', y=y, x='layer', style='Processing condition', markers=True, palette= representation_colors, ax=line_plot_ax);
        # line_plot_ax.set_xticks(model_df.layer.unique());
        line_plot_ax.set_xticks([layer for layer in model_df.layer.unique() if layer % 4 == 0])


#         g.set(ylim=)
        g.set(ylim=ylim);

        if fig_x == 0:
            g.set_ylabel('Correlation');
        else:
            g.set_ylabel('');
            
        if fig_y == 0:
            g.set_xlabel('');

        # if fig_x == 0 and fig_y == 0:
        #     g.legend(loc='best', bbox_to_anchor=(1.5, -2, 0.5, 0.5), ncol=1);
        # else:
        #     g.legend().remove();
        existing_handles, existing_labels = g.get_legend_handles_labels()
        legend = g.get_legend()
        if fig_x == 0 and fig_y == 0:
            g.legend(
                existing_handles + sig_legend_handles,
                existing_labels + [h.get_label() for h in sig_legend_handles],
                loc='upper right',
                prop={'size': 7},
                markerscale=0.75
            )
        elif legend is not None:
            legend.remove()

        
        g.axhline(0, color='black', linestyle='--', linewidth=1)
        
        g.set_title(model_name);
        
        # for layer in model_df.layer.unique():
        #     pval = paired_t_test_df[(paired_t_test_df.model == model) & (paired_t_test_df.layer == layer)]['corrected_p'].iloc[0]
        #     x_offset =  0.075 if 'distil' in model else 0.2
        #     y_offset = 0.1 - 0.035

        #     if pval <= 0.05:
        #         y_val = model_df[(model_df.layer==layer)]['same_relation_group_rdm_corr'].min()
        #         plt.text(layer - x_offset, 0 - y_offset, '*', weight='bold', size='x-large', color='black')
        for layer in model_df.layer.unique():
            # Existing Comparison (Figurative vs Standard)
            pval1 = paired_t_test_df[(paired_t_test_df.model == model) & (paired_t_test_df.layer == layer)]['corrected_p'].iloc[0]
            
            # New Comparison (Figurative vs No Context)
            pval2 = paired_t_test_context_vs_no_df[(paired_t_test_context_vs_no_df.model == model) & (paired_t_test_context_vs_no_df.layer == layer)]['corrected_p'].iloc[0]
            
            x_offset = 0.2
            y_base = -0.05 # Adjust based on your ylim
            
            # Draw Asterisk 1: Figurative vs Standard (Green color to match 'Standard' line?)
            if 0 <= pval1 <= 0.05:
                plt.text(layer - x_offset, y_base, '*', weight='bold', size='large', color='#6C757D')
            
            # Draw Asterisk 2: Figurative vs No Context (Red color to match 'Literal' line?)
            # Positioned slightly lower
            if 0 <= pval2 <= 0.05:
                plt.text(layer - x_offset, y_base - 0.05, '*', weight='bold', size='large', color='#E03131')
        
    fig.tight_layout()
    fig.show()

plt.savefig('figures/idioms_context_comparison_v2_with_mask.png', format = 'png')
plt.savefig('figures/idioms_context_comparison_v2_with_mask.eps', format='eps')