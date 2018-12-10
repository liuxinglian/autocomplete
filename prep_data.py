# prep_data.py

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