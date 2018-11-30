from pathlib import Path
import json
import numpy as np
import nltk
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec


def preprocess_json_data(filename):
    with open(filename) as f:
        data = f.readlines()
    reviews = [json.loads(x.strip()) for x in data]
    # print(reviews[0]['text'])
    sentences = [nltk.word_tokenize(reviews[i]['text']) for i in range(5000)]

    saved_model = my_file = Path('model.txt')

    if not saved_model.is_file():
        model = Word2Vec(sentences, workers=4, sg=1, min_count=1)
        learned_vocab = list(model.wv.vocab)
        
        # save the trained model
        model.wv.save_word2vec_format('model.txt')
        saved_model = my_file = Path('model.txt')

    else:
        model = word2vec.KeyedVectors.load_word2vec_format('model.txt', binary=False)

    # print(model['pizza'])
    print(list(model.wv.vocab))
    print(model.most_similar(positive=[model['amazing']], topn=3))

    # vocab = set()
    # for review in reviews:
    #     vocab = vocab|set(review['text'].split())

    # print(len(vocab))
    # print(list(vocab)[0])

def main():
    preprocess_json_data('yelp_academic_dataset_review.json')

if __name__ == '__main__':
    main()