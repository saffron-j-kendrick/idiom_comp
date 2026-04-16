import pandas as pd
import os
import numpy as np
import tqdm
import torch

import rsa_utils


def load_correct_form_standard(data_loc='data'):
    df = pd.read_excel('{}/correct_form_standard.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_standard_and(data_loc='data'):
    df = pd.read_excel('{}/correct_form_standard.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))

def load_correct_form_standard_bs(data_loc='data'):
    df = pd.read_excel('{}/correct_form_standard.xlsx'.format(data_loc))
    return list(zip(df.b1_match.tolist(), df.b2_match.tolist()))

def get_standard_sentences(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_standard(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_standard_and(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_standard_bs(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['b1'].tolist(), df['b2'].tolist())))
    return mod_head_tuples_per_sentence



def load_correct_form_context(data_loc='data'):
    df = pd.read_excel('{}/correct_form_context.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_context_and(data_loc='data'):
    df = pd.read_excel('{}/correct_form_context.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))

def load_correct_form_context_bs(data_loc='data'):
    df = pd.read_excel('{}/correct_form_context.xlsx'.format(data_loc))
    return list(zip(df.b1_match.tolist(), df.b2_match.tolist()))

def get_context_sentences(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences_context.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_context(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences_context.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_context_and(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences_context.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_context_bs(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences_context.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['b1'].tolist(), df['b2'].tolist())))
    return mod_head_tuples_per_sentence



def load_correct_form_no_context(data_loc='data'):
    df = pd.read_excel('{}/correct_form_no_context.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_no_context_and(data_loc='data'):
    df = pd.read_excel('{}/correct_form_no_context.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))

def load_correct_form_no_context_bs(data_loc='data'):
    df = pd.read_excel('{}/correct_form_no_context.xlsx'.format(data_loc))
    return list(zip(df.b1_match.tolist(), df.b2_match.tolist()))

def get_no_context_sentences(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences_no_context.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_no_context(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences_no_context.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_no_context_and(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences_no_context.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_no_context_bs(data_loc='data'):
    df = pd.read_excel('{}/standard_sentences_no_context.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['b1'].tolist(), df['b2'].tolist())))
    return mod_head_tuples_per_sentence










def load_correct_form_short(data_loc='data'):
    df = pd.read_excel('{}/correct_form_short.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_short_and(data_loc='data'):
    df = pd.read_excel('{}/correct_form_short.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))

def load_correct_form_short_bs(data_loc='data'):
    df = pd.read_excel('{}/correct_form_short.xlsx'.format(data_loc))
    return list(zip(df.b1_match.tolist(), df.b2_match.tolist()))


def get_short_sentences(data_loc='data'):
    df = pd.read_excel('{}/context_short.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_short(data_loc='data'):
    df = pd.read_excel('{}/context_short.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_short_and(data_loc='data'):
    df = pd.read_excel('{}/context_short.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_short_bs(data_loc='data'):
    df = pd.read_excel('{}/context_short.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['b1'].tolist(), df['b2'].tolist())))
    return mod_head_tuples_per_sentence


def load_correct_form_single(data_loc='data'):
    df = pd.read_excel('{}/correct_form_literal_single.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_single_and(data_loc='data'):
    df = pd.read_excel('{}/correct_form_literal_single.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))

def load_correct_form_single_bs(data_loc='data'):
    df = pd.read_excel('{}/correct_form_literal_single.xlsx'.format(data_loc))
    return list(zip(df.b1_match.tolist(), df.b2_match.tolist()))


def get_single_sentences(data_loc='data'):
    df = pd.read_excel('{}/literal_context_single.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_single(data_loc='data'):
    df = pd.read_excel('{}/literal_context_single.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_single_and(data_loc='data'):
    df = pd.read_excel('{}/literal_context_single.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_single_bs(data_loc='data'):
    df = pd.read_excel('{}/literal_context_single.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['b1'].tolist(), df['b2'].tolist())))
    return mod_head_tuples_per_sentence





def load_correct_form_single_no_context(data_loc='data'):
    df = pd.read_excel('{}/correct_form_no_context_single.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_single_no_context_and(data_loc='data'):
    df = pd.read_excel('{}/correct_form_no_context_single.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))

def load_correct_form_single_no_context_bs(data_loc='data'):
    df = pd.read_excel('{}/correct_form_no_context_single.xlsx'.format(data_loc))
    return list(zip(df.b1_match.tolist(), df.b2_match.tolist()))


def get_single_sentences_no_context(data_loc='data'):
    df = pd.read_excel('{}/no_context_single.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_single_no_context(data_loc='data'):
    df = pd.read_excel('{}/no_context_single.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_single_no_context_and(data_loc='data'):
    df = pd.read_excel('{}/no_context_single.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_single_no_context_bs(data_loc='data'):
    df = pd.read_excel('{}/no_context_single.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['b1'].tolist(), df['b2'].tolist())))
    return mod_head_tuples_per_sentence

# def load_correct_form_idioms_per_sentence(data_loc='data'):
#     df = pd.read_excel('{}/correct_idiom_form_copy.xlsx'.format(data_loc))
#     return list(zip(df.modifier_match.tolist(), df.head_match.tolist()))

# def load_correct_form_idioms_per_sentence_with_and(data_loc='data'):
#     df = pd.read_excel('{}/correct_idiom_form_copy_and_word.xlsx'.format(data_loc))
#     return list(zip(df.head_match.tolist(), df.and_match.tolist()))


# def load_correct_form_idioms_per_sentence_with_context(data_loc='data'):
#     df = pd.read_excel('{}/correct_idiom_form_copy_with_context.xlsx'.format(data_loc))
#     return list(zip(df.modifier_match.tolist(), df.head_match.tolist()))

# def load_correct_form_idioms_per_sentence_with_context_and(data_loc='data'):
#     df = pd.read_excel('{}/correct_idiom_form_copy_and_word_with_context.xlsx'.format(data_loc))
#     return list(zip(df.head_match.tolist(), df.and_match.tolist()))



def get_hidden_state_file(model_name, layer=11, rep_type='sentence_pair_cls', data_loc='data'):
    hidden_state_folder = '{}/representations/{}/layer_{}/{}'.format(data_loc, model_name.split('-')[0], layer, rep_type)
    return '{}/{}_layer_{}_{}.npy'.format(hidden_state_folder, model_name, layer, rep_type)


# def get_idiom_test_sentences(data_loc='data'):
#     df = pd.read_excel('{}/idiom_test_sentences_copy.xlsx'.format(data_loc))
#     sentences = np.array(df['sentence'].tolist())
#     return sentences

# def get_idiom_test_sentences_with_context(data_loc='data'):
#     df = pd.read_excel('{}/idiom_test_sentences_copy_with_context.xlsx'.format(data_loc))
#     sentences = np.array(df['sentence'].tolist())
#     return sentences

# def get_idiom_modifier_head_words_per_sentence(data_loc='data'):
#     df = pd.read_excel('{}/idiom_test_sentences_copy.xlsx'.format(data_loc))
#     mod_head_tuples_per_sentence = np.array(list(zip(df['modifier'].tolist(), df['head'].tolist())))
#     return mod_head_tuples_per_sentence

# def get_idiom_modifier_head_words_per_sentence_with_context(data_loc='data'):
#     df = pd.read_excel('{}/idiom_test_sentences_copy_with_context.xlsx'.format(data_loc))
#     mod_head_tuples_per_sentence_with_context = np.array(list(zip(df['modifier'].tolist(), df['head'].tolist())))
#     return mod_head_tuples_per_sentence_with_context

# def get_idiom_modifier_head_words_per_sentence_with_and(data_loc='data'):
#     df = pd.read_excel('{}/idiom_test_sentences_copy.xlsx'.format(data_loc))
#     and_head_tuples_per_sentence = np.array(list(zip(df['head'].tolist(), df['and'].tolist())))
#     return and_head_tuples_per_sentence

# def get_idiom_modifier_head_words_per_sentence_with_context_and(data_loc='data'):
#     df = pd.read_excel('{}/idiom_test_sentences_copy_with_context.xlsx'.format(data_loc))
#     and_head_tuples_per_sentence_with_context = np.array(list(zip(df['head'].tolist(), df['and'].tolist())))
#     return and_head_tuples_per_sentence_with_context

def select_within_compound_groups(rdm, group_i):
    to_keep_inds = []
    
    get_lower = lambda x: x[np.where(np.triu(np.ones(x.shape[:1])) == 0)]

    for start in range(0, 320, 8):  # Groups of 16×16
        block_inds = [[(i, j) for i in range(start, start + 8)] for j in range(start, start + 8)]
        to_keep_inds.append(get_lower(np.array(block_inds)))

    return np.array([rdm[i[0]][i[1]] for i in to_keep_inds[group_i]])
