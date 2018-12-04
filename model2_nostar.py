from pathlib import Path
from operator import itemgetter
import json, sys
import shutil, os
import numpy as np
import nltk
import tensorflow as tf
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec

SAVE_PATH = './model/m.cpkt'

def get_review_data(filename, start, end):
    with open(filename) as f:
        data = f.readlines()
    reviews = [json.loads(x.strip()) for x in data]
    # print(reviews[0]['text'])
    sentences = [nltk.word_tokenize(reviews[i]['text'].lower()) for i in range(start, end)]

    return sentences


# return word2Vec model that can extract word embedding
def get_word_embedding(filename, start_train, end_train):
    train_size = end_train - start_train
    path = 'model_' + str(train_size) + '.txt'
    sentences = get_review_data(filename, start_train, end_train)
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

    return model, sentences


# start predicting from the 6th word
def prepare_input_for_nn(model, sentences):
    # list of numpy array (each is a embedding representing the previous sequence)
    inputs = []
    # list of vector (true word representing by vector)
    true_words = []
    for sentence in sentences:
        if len(sentence) < 5:
            continue

        # unnormalized one doesn't divide by the total weight
        weighted_sum = np.zeros(model.vector_size)
        total_weight = 0
        for i in range(5):
            total_weight += i+1
            if sentence[i] in model.wv.vocab:
                weighted_sum += model[sentence[i]] * (i+1)

        # begin prepare input and label
        for i in range(5, len(sentence)):
            cur_input = weighted_sum / total_weight
            cur_label = sentence[i]

            if cur_label in model.wv.vocab:
                inputs.append(cur_input)
                true_words.append(model[cur_label])

                weighted_sum += model[cur_label] * (i+1)

            total_weight += i+1
            

    return inputs, true_words


def build_nn(input_ph, out_size=100):
    hidden1 = tf.layers.dense(inputs=input_ph, units=128, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(inputs=hidden1, units=256, activation=tf.nn.relu)
    hidden3 = tf.layers.dense(inputs=hidden2, units=128, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=hidden3, units=out_size)
    return output

def get_loss(pred_word, true_word):
    # consine distance
    loss = tf.losses.cosine_distance(tf.nn.l2_normalize(pred_word, 0), tf.nn.l2_normalize(true_word, 0), dim=0)
    return loss

def get_optimizer(loss, lr=0.005):
    # return a tf operation
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


def train_nn(model, sess, saver, input_ph, word_ph, loss, train_op, inputs, true_words, batch_size, training, num_epoch):
    print("begin training")
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    loss_summary = tf.summary.scalar('loss', loss)
    # merged = tf.summary.merge_all()
    index = np.arange(len(inputs))
    # change inputs and true_words vectors to np array
    ### inputs = np.reshape(np.array(inputs), (len(inputs), model.vector_size))
    ### true_words = np.reshape(np.array(true_words), (len(inputs), model.vector_size))

    iterations = int(len(inputs)/batch_size)

    for i in range(iterations):
        batch_index = np.random.choice(index, size=batch_size, replace=True)
        batch_inputs = itemgetter(*batch_index)(inputs)
        batch_words = itemgetter(*batch_index)(true_words)
        batch_inputs_np = np.reshape(np.array(batch_inputs), (batch_size, model.vector_size))
        batch_words_np = np.reshape(np.array(batch_words), (batch_size, model.vector_size))
        sess.run(train_op, feed_dict={input_ph: batch_inputs_np, word_ph: batch_words_np, training:False})
        cur_loss = sess.run(loss, feed_dict={input_ph: batch_inputs_np, word_ph: batch_words_np, training:False})
        # print("loss for batch {} is {}".format(i, cur_loss))
        summary = sess.run(loss_summary, feed_dict={input_ph: batch_inputs_np, word_ph: batch_words_np, training:False})
        writer.add_summary(summary, i)


    for r in range(num_epoch-1):
        for i in range(iterations):
            batch_index = np.random.choice(index, size=batch_size, replace=True)
            batch_inputs = itemgetter(*batch_index)(inputs)
            batch_words = itemgetter(*batch_index)(true_words)
            batch_inputs_np = np.reshape(np.array(batch_inputs), (batch_size, model.vector_size))
            batch_words_np = np.reshape(np.array(batch_words), (batch_size, model.vector_size))
            sess.run(train_op, feed_dict={input_ph: batch_inputs_np, word_ph: batch_words_np, training:False})
            cur_loss = sess.run(loss, feed_dict={input_ph: batch_inputs_np, word_ph: batch_words_np, training:False})
            # print("loss for batch {} is {}".format(i, cur_loss))

    
    saver.save(sess, SAVE_PATH)


def get_prediction(model, nn_model, test_sentences, input_ph, word_ph, training_ph):
    print('begin predicting')
    test_inputs, test_true_words = prepare_input_for_nn(model, test_sentences)
    print('test true word len = {}'.format(len(test_true_words)))
    test_inputs = np.reshape(np.array(test_inputs), (len(test_inputs), model.vector_size))
    test_true_words = np.reshape(np.array(test_true_words), (len(test_true_words), model.vector_size))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_PATH)
        test_pred_words = sess.run(nn_model, feed_dict={input_ph: test_inputs, word_ph: test_true_words, training_ph:False})
    print('test pred word len = {}'.format(len(test_pred_words)))
    return test_true_words, test_pred_words


def get_accuracy(model, true_words, pred_words, topn=10):
    print('begin getting accuracy')
    correct = 0
    for i in range(len(true_words)):
        if (i % 100 == 0):
            print("calc acc : {} done".format(i))
        true_word_vec = true_words[i]
        pred_word_vec = pred_words[i]

        temp1 = model.most_similar([true_word_vec], [], topn)
        true_words_set = set([pair[0] for pair in temp1])
        temp2 = model.most_similar([pred_word_vec], [], topn)
        pred_words_set = set([pair[0] for pair in temp2])

        if bool(true_words_set & pred_words_set):
            correct += 1

    return correct / len(true_words)


def main(start_train, end_train, start_test, end_test, epoch):
    if len(sys.argv) == 7:
        if os.path.isdir("model"):
            shutil.rmtree('model')
        if os.path.isdir("graphs"):
            shutil.rmtree('graphs')
    model, sentences = get_word_embedding('yelp_academic_dataset_review.json', start_train, end_train)
    train_fea, train_label = prepare_input_for_nn(model, sentences)
    test_sentences = get_review_data('yelp_academic_dataset_review.json', start_test, end_test)
    print("----------------------- DONE WITH GET REVIEW DATA -----------------------")
    input_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='train_input')
    word_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='train_label')
    training = tf.placeholder(tf.bool)
    nn_model = build_nn(input_ph)
    loss = get_loss(nn_model, word_ph)
    train_op = get_optimizer(loss, 0.001)
    saver = tf.train.Saver()
    # begin training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        train_nn(model, sess, saver, input_ph, word_ph, loss, train_op, train_fea, train_label, 128, training, num_epoch=epoch)
    print("----------------------- DONE WITH TRAINING -----------------------")
    # t_input_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='test_input')
    # t_word_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='test_predicted_label')
    test_true_words, test_pred_words = get_prediction(model, nn_model, test_sentences, input_ph, word_ph, training)
    print("----------------------- DONE WITH PREDICTION -----------------------")
    acc = get_accuracy(model, test_true_words, test_pred_words)
    print("----------------------- DONE WITH GET ACCURACY -----------------------")
    print('accuracy = {}'.format(acc))

if __name__ == '__main__':
    start_train = int(sys.argv[1])
    end_train = int(sys.argv[2])
    start_test = int(sys.argv[3])
    end_test = int(sys.argv[4])
    epoch = int(sys.argv[5])
    main(start_train, end_train, start_test, end_test, epoch)
