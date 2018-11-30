from pathlib import Path
import json
import numpy as np
import nltk
import tensorflow as tf
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec


# return word2Vec model that can extract word embedding
def get_word_embedding(filename):

    sentences = get_review_data(filename)
    saved_model = my_file = Path('model.txt')

    if not saved_model.is_file():
        
        model = Word2Vec(sentences, size=100, workers=4, sg=1, min_count=1)
        
        # save the trained model
        model.wv.save_word2vec_format('model.txt')
        saved_model = my_file = Path('model.txt')

    else:
        model = word2vec.KeyedVectors.load_word2vec_format('model.txt', binary=False)

    learned_vocab = list(model.wv.vocab)
    # print(model['pizza'])
    # print(list(learned_vocab))
    print(model.most_similar(positive=[model['pizza']], topn=3))

    return model, stentences


def get_review_data(filename, num_reviews=5000):
    with open(filename) as f:
        data = f.readlines()
    reviews = [json.loads(x.strip()) for x in data]
    # print(reviews[0]['text'])
    sentences = [nltk.word_tokenize(reviews[i]['text']) for i in range(num_reviews)]

    return sentences


# fixed length for now, length = 8, use previous 7 words to predict the 8th word
def prepare_input_for_nn(model, sentences):
    # list of numpy array (each is a embedding representing the previous sequence)
    inputs = []
    # list of string
    true_words = []
    for sentence in sentences:
        for i in range(len(sentence)-7):
            cur = np.zeros(model.vector_size)
            for j in range(1, 8):
                cur += j * model[sentence[i+j-1]]

            inputs.append(cur)
            true_words.append(sentence[i+7])

    return inputs, true_words


def build_nn():
    pass


def main():
    model, sentences = get_word_embedding('yelp_academic_dataset_review.json')
    train_fea, train_label = prepare_input_for_nn(model, stentences)

if __name__ == '__main__':
    main()