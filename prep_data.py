# prep_data.py

from pathlib import Path
from operator import itemgetter
import json, sys, shutil, os
import numpy as np
import nltk
import tensorflow as tf
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec

def get_review_data(filename, start, end):
    with open(filename) as f:
        data = f.readlines()
    reviews = [json.loads(x.strip()) for x in data]
    sentences = []
    stars = []
    # sentences = [nltk.word_tokenize(reviews[i]['text'].lower()) for i in range(start, end)]
    for i in range(start, end):
        sentences.append(nltk.word_tokenize(reviews[i]['text'].lower()))
        stars.append(reviews[i]['stars'])

    return sentences, stars

# return word2Vec model that can extract word embedding
def get_word_embedding(filename, start_train, end_train):
    train_size = end_train - start_train
    path = 'model_' + str(train_size) + '.txt'
    sentences, stars = get_review_data(filename, start_train, end_train)
    saved_model = my_file = Path(path)

    if not saved_model.is_file():
        
        model = Word2Vec(sentences, size=100, workers=8, sg=1, min_count=1)
        
        # save the trained model
        model.wv.save_word2vec_format(path)
        saved_model = my_file = Path(path)

    else:
        model = word2vec.KeyedVectors.load_word2vec_format(path, binary=False)

    learned_vocab = list(model.wv.vocab)
    # print(model['pizza'])
    # print(list(learned_vocab))
    # print(model.most_similar(positive=[model['pizza']], topn=3))

    return model, sentences, stars


# Reference: https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
