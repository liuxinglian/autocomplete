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

import warnings
warnings.filterwarnings('ignore', '.*do not.*',)

tf.reset_default_graph()

with tf.Session() as sess:
    SAVE_PATH = './model/m.cpkt'
    saver = tf.train.import_meta_graph('./model/m.cpkt.meta')
    saver.restore(sess, SAVE_PATH)

    while True:
        rate = input("Please give a rating in the scale of 5:\n")
        rate = int(rate)
        assert(rate >= 1 and rate <= 5)
        line = input("Please give some input\n")
        sentence = line.split()
        sentence_embedding = []

        path = 'model_10000.txt'
        word2vec_model = word2vec.KeyedVectors.load_word2vec_format(path, binary=False)

        for i in range(len(sentence)):
            if sentence[i] in word2vec_model.wv.vocab:
                sentence_embedding.append(word2vec_model[sentence[i]])
            else:
                sentence_embedding.append(np.zeros(word2vec_model.vector_size, dtype=np.float32))
        sentence_embedding = np.array(sentence_embedding)

        # begin prepare input
        n_steps = 20
        reverse = True
        if len(sentence) < n_steps + 1:
            cur_input = sentence_embedding[:-1]
            pad_num = n_steps - len(sentence) + 1
            pad = np.zeros((pad_num, word2vec_model.vector_size))
            cur_input = np.concatenate((pad, cur_input), axis=0) 
        else:
            cur_input = sentence_embedding[len(sentence)-num_steps:-1]
        if reverse:
            cur_input=np.flip(cur_input,0)
            
        nn_model = tf.get_default_graph().get_tensor_by_name("dense_3/BiasAdd:0")
        input_ph = tf.get_default_graph().get_tensor_by_name("train_input:0")
           
        pred = sess.run(nn_model, feed_dict={input_ph: cur_input, word_ph: np.zeros((1,100), dtype=np.float32), training_ph:False})        
        pred_word = pred_dict_filter(word2vec_model, sentence[-1], pred[-1], topn=1, cons=20)
        print(pred_word)
