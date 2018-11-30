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


# start predicting from the 6th word
def prepare_train_input_for_nn(model, train_sentences):
    # list of numpy array (each is a embedding representing the previous sequence)
    inputs = []
    # list of string
    true_words = []
    for sentence in train_sentences:
        if len(sentence) < 5:
            continue

        # unnormalized one doesn't divide by the total weight
        weighted_sum = np.zeros(model.vector_size)
        total_weight = 0
        for i in range(5):
            total_weight += i+1
            weighted_sum += model[sentence[i]] * (i+1)

        # begin prepare input and label
        for i in range(5, len(sentence)):
            cur_input = weighted_sum / total_weight
            cur_label = sentence[i]
            inputs.append(cur_input)
            true_words.append(cur_label)

            weighted_sum += model[sentence[i]] * (i+1)
            total_weight += i+1
            

    return inputs, true_words


def build_nn():
    pass

def get_loss(pred_word, true_word):
    # consine distance
    loss = tf.losses.cosine_distance(tf.nn.l2_normalize(pred_word, 0), tf.nn.l2_normalize(true_word, 0), dim=0)
    return loss

def get_optimizer(loss, lr=0.005):
    # return a tf operation
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


def train_nn(model, sess, input_ph, word_ph, loss, train_op, inputs, true_words, batch_size):
    index = np.arange(len(inputs))
    # change inputs and true_words vectors to np array
    inputs = np.reshape(np.array(inputs), (len(inputs), model.vector_size))
    true_words = np.reshape(np.array(true_words))

    iterations = int(len(inputs)/batch_size)

    for i in range(iterations):
        batch_index = np.random.choice(index, size=batch_size, replace=True)
        batch_inputs = inputs[batch_index]
        batch_words = true_words[batch_index]
        sess.run(train_op, feed_dict={input_ph: batch_inputs, word_ph: batch_words, training: True})
        cur_loss = sess.run(loss, feed_dict={input_ph: batch_inputs, word_ph: batch_words, training: True})
        print("loss for batch {} is {}".format(i, cur_loss))


def main():
    model, sentences = get_word_embedding('yelp_academic_dataset_review.json')
    train_fea, train_label = prepare_input_for_nn(model, stentences)

    input_ph = tf.placeholder(tf.float32, [None, model.vector_size])
    word_ph = tf.placeholder(tf.float32, [None])
    training = tf.placeholder(tf.bool)
    nn_model = build_nn()
    loss = get_loss(nn_model, word_ph)
    train_op = optimizer(loss)

    # begin training
    with tf.Session() as sess:
        print("begin training")
        train_cnn(model, sess, input_ph, word_ph, loss, train_op, inputs, true_words, 32)

if __name__ == '__main__':
    main()
