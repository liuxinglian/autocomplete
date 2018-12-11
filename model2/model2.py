from pathlib import Path
from operator import itemgetter
import json, sys, shutil, os
import numpy as np
import nltk
import tensorflow as tf
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
from model2_config import model2_params

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system_config import system_params
from prep_data import get_review_data, get_word_embedding

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# start predicting from the 6th word
def prepare_input_for_nn(model, sentences, stars, reverse=False):
    # list of numpy array (each is a embedding representing the previous sequence)
    inputs = []
    # list of vector (true word representing by vector)
    true_words = []
    if not reverse:
        for i in range(len(sentences)):
            sentence = sentences[i]
            star = np.array([stars[i]])
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

    else:
        max_weight = len(sentences)
        for i in range(len(sentences)):
            sentence = sentences[i]
            star = np.array([stars[i]])
            if len(sentence) < 5:
                continue

            # unnormalized one doesn't divide by the total weight
            weighted_sum = np.zeros(model.vector_size)
            total_weight = 0
            for i in range(5):
                cur_weight = max_weight-i-1
                total_weight += cur_weight
                if sentence[i] in model.wv.vocab:
                    weighted_sum += model[sentence[i]] * (cur_weight)

            # begin prepare input and label
            for i in range(5, len(sentence)):
                cur_weight = max_weight-i-1
                cur_input = weighted_sum / total_weight
                cur_label = sentence[i]

                if cur_label in model.wv.vocab:
                    inputs.append(cur_input)
                    true_words.append(model[cur_label])

                    weighted_sum += model[cur_label] * cur_weight

                total_weight += cur_weight

            

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
    # loss = tf.losses.softmax_cross_entropy(true_word, pred_word)
    return loss

def get_optimizer(loss, lr=0.005):
    # return a tf operation
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

def train_nn(model, sess, saver, input_ph, word_ph, loss, train_op, inputs, true_words, batch_size, training, num_epoch, save_path):
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
        if i % 1000 == 0:
            print("loss for batch {} is {}".format(i, cur_loss))
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
            if i % 1000 == 0:
                print("loss for batch {} is {}".format(i, cur_loss))

    
    saver.save(sess, save_path)


def get_prediction(model, nn_model, test_sentences, test_stars, input_ph, word_ph, training_ph, save_path):
    print('begin predicting')
    test_inputs, test_true_words = prepare_input_for_nn(model, test_sentences, test_stars, reverse=False)
    print('test true word len = {}'.format(len(test_true_words)))
    test_inputs = np.reshape(np.array(test_inputs), (len(test_inputs), model.vector_size))
    test_true_words = np.reshape(np.array(test_true_words), (len(test_true_words), model.vector_size))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
        test_pred_words = sess.run(nn_model, feed_dict={input_ph: test_inputs, word_ph: test_true_words, training_ph:False})
    print('test pred word len = {}'.format(len(test_pred_words)))
    return test_true_words, test_pred_words


def get_accuracy(model, true_words, pred_words, topn=10):
    print('begin getting accuracy')
    correct = 0
    for i in range(len(true_words)):

        true_word_vec = true_words[i]
        pred_word_vec = pred_words[i]

        temp1 = model.most_similar([true_word_vec], [], topn)
        true_words_set = set([pair[0] for pair in temp1])
        temp2 = model.most_similar([pred_word_vec], [], topn)
        pred_words_set = set([pair[0] for pair in temp2])

        if bool(true_words_set & pred_words_set):
            correct += 1

    return correct / len(true_words)


def main():
    sys_params = system_params()
    model_params = model2_params()

    save_path = model_params.tf_save_path
    save_folder = os.path.dirname(save_path)
    while os.path.isdir(save_folder):
        overwrite = input("There is a existing model on this path, overwrite? [y/n]")
        if (overwrite == 'y'):
            shutil.rmtree(save_folder)

    start_train, end_train = model_params.train_start, model_params.train_end
    start_test, end_test = model_params.test_start, model_params.test_end
    
    print('---------------- Getting Data ----------------')
    wv_model, train_sentences, train_stars = get_word_embedding(sys_params.all_reviews_jsonfn, start_train, end_train)
    test_sentences, test_stars = get_review_data(sys_params.all_reviews_jsonfn, start_test, end_test)
    print('---------------- Done Getting Data ----------------')

    print('---------------- Prepaing Input for Neural Network ----------------')
    train_fea, train_label = prepare_input_for_nn(wv_model, train_sentences, train_stars, reverse=False)
    print('---------------- Done Prepaing Input for Neural Network ----------------')
    
    input_ph = tf.placeholder(tf.float32, [None, wv_model.vector_size], name='train_input')
    word_ph = tf.placeholder(tf.float32, [None, wv_model.vector_size], name='train_label')
    training = tf.placeholder(tf.bool)
    
    nn_model = build_nn(input_ph)
    loss = get_loss(nn_model, word_ph)
    train_op = get_optimizer(loss, model_params.learning_rate)
    saver = tf.train.Saver()

    # begin training
    print("---------------- Training ----------------")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        train_nn(wv_model, sess, saver, input_ph, word_ph, loss, train_op, train_fea, train_label, 32, training, num_epoch=model_params.epoches, save_path=save_path)
    print("---------------- Done Training ----------------")
    # t_input_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='test_input')
    # t_word_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='test_predicted_label')
    print("---------------- Predicting ----------------")
    test_true_words, test_pred_words = get_prediction(wv_model, nn_model, test_sentences, test_stars, input_ph, word_ph, training, save_path)
    print("---------------- Done Predicting ----------------")
    print("---------------- Getting Accuracy ----------------")
    acc = get_accuracy(wv_model, test_true_words, test_pred_words)
    print('Accuracy is {}'.format(acc))

if __name__ == '__main__':
    main()
