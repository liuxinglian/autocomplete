from pathlib import Path
from operator import itemgetter
import json, sys
import shutil, os
import numpy as np
import nltk
import tensorflow as tf
import pygtrie as trie
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec


'''
This is a file that evaluates the model using query autocompletion metrics.
get_esaved function will return average esaved for the model.
dict_filter will return a word which is the closest prediction given the inputs.
'''

def get_esaved(model, true_words, pred_words, topn=1, cons=20):
    '''
    This function evaluates the model by calculating the eSaved as defined in checkpoint2.
    Inputs:
        model: the word2vec model precalculated
        true_words: the true words in the original text
        pred_words: the word vecs produced by the model
        topn: the number of nearest neighbours to be evaluated as correct, default to 1
        cons: the number of nearest neighbours to be considered, default to 20
    Output:
    The average eSaved.
    '''
    print('begin getting eSaved')
    
    # calculate the esaved
    eSaved = 0
    for i in range(len(true_words)):

        true_word_vec = true_words[i]
        pred_word_vec = pred_words[i]

        temp1 = model.most_similar([true_word_vec], [], 1)
        true_word = temp1[0][0]
        similar_list = model.most_similar([pred_word_vec], [], cons)
        pred_words_list = [pair[0] for pair in similar_list]
        # build a trie tree to speed up the finding process
        t = trie.CharTrie()
        rt = {}
        for i in range(len(pred_words_list)):
            word = pred_words_list[i]
            t[word] = i
            rt[i] = word
        inputs = ''
        flag = False
        for i in range(len(true_word)):
            inputs += true_word[i]
            es = 1 - (i + 1) / (len(true_word) + 1)
            try:
                m = list(t[inputs:])
                m.sort()
                for j in range(min(topn, len(m))):
                    if rt[m[j]] == true_word:
                        eSaved += es
                        flag = True
                        break
            except KeyError:
                break
            if flag:
                break
    return eSaved / len(true_words)


def pred_dict_filter(model, inputs, pred_word_vec, topn=1, cons=20):
    '''
    This function gives a dict filter to our predictions.
    Inputs:
        model: the word2vec model precalculated
        inputs: the input word
        pred_word_vec: the word vec produced by the model
        topn: the number of nearest neighbours to be evaluated as correct, default to 1
        cons: the number of nearest neighbours to be considered, default to 20
    Output:
        The predicted word.
    '''
    similar_list = model.most_similar([pred_word_vec], [], cons)
    pred_words_list = [pair[0] for pair in similar_list]
    '''
    # This is a deprecated version that use trie tree to do the whole process
    # build a trie tree to speed up the finding process
    t = trie.CharTrie()
    rt = {}
    for i in range(len(pred_words_list)):
        word = pred_words_list[i]
        t[word] = i
        rt[i] = word
    try:
        m = list(t[inputs:])
        m.sort()
        pred_word = rt[m[0]]
    except KeyError:
        return ''
    return pred_word
    '''
    # This is a better version that uses Python's substring method
    for i in range(len(pred_words_list)):
        word = pred_words_list[i]
        if len(inputs) > len(word):
            continue
        if inputs == word[0:len(inputs)]:
            return word
    return ''
