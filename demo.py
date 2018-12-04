from pathlib import Path
from operator import itemgetter
import json, sys
import shutil, os
import numpy as np
import nltk
import tensorflow as tf
import pygtrie as trie
from dict_filter import pred_dict_filter
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec


'''
This is a file that evaluates the model using query autocompletion metrics.
get_esaved function will return average esaved for the model.
dict_filter will return a word which is the closest prediction given the inputs.
'''

def demo(w2v_model, pred_model, topn=1, n_steps=20, reverse=True):
    '''
    Provides an interactive interface that generates prediction.
    Inputs:
        w2v_model: pretrained word2vec model
        pred_model: pretrained prediction model as proposed in project: LM or NN or LSTM
        topn: top n predictions to be generated
        n_steps: number of previous words to be considered
        reverse: if the input should be reversed
    '''
    while True:
        print("Please give a rating in the scale of 5:")
        rate = sys.stdin.readline()
        rate = int(rate*2)/2
        print("Please give some input")
        line = sys.stdin.readline()
        sentence = line.split()
        sentence_embedding = []
        for i in range(len(sentence)):
            if sentence[i] in model.wv.vocab:
                sentence_embedding.append(model[sentence[i]])
            else:
                sentence_embedding.append(np.zeros(model.vector_size, dtype=np.float32))
        sentence_embedding = np.array(sentence_embedding)
        # begin prepare input
        if len(sentence) < n_steps + 1:
            cur_input = sentence_embedding[:-1]
            pad_num = n_steps - len(sentence)
            pad = np.zeros((pad_num, model.vector_size))
            cur_input = np.concatenate((pad, cur_input), axis=0) 
        else:
            cur_input = sentence_embedding[len(sentence)-num_steps:-1]
        if reverse:
            cur_input=np.flip(cur_input,0)
        pred = pred_model(cur_input)
        pred_word = dict_filter(model, sentence[-1], pred, topn=1, cons=20)
        print(pred_word)



if __name__ == "__main__":
    w2v_model = some_model_restore_a()
    pred_model = some_model_restore_b()
    demo(w2v_model, pred_model, topn=1, n_steps=20, reverse=True)