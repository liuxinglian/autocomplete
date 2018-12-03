from pathlib import Path
from operator import itemgetter
import json
import numpy as np
import nltk
import tensorflow as tf
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec

SAVE_PATH = './model/m.cpkt'


class DataSet(object):
    '''
    This is an object that is holding the data and generate batches.
    '''
    def __init__(self, 
                 data,
                 label,
                 seq_length,
                 training=True):
        self.data = data
        self.label = label
        self.seq_length = seq_length
        self.training = training
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_data = data.shape[0]
        self._curr_order = np.arange(self._num_data)

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        #print(self.training)
        if self.training == False:
            #print("triggered")
            return self.data, self.label, self.seq_length
        if start == 0 and self._epochs_completed == 0 and shuffle:
            np.random.shuffle(self._curr_order)
        if start + batch_size > self._num_data:
            self._epochs_completed += 1
            num_feeded = self._num_data - start
            num_rest = batch_size - num_feeded
            X_batch_feeded = self.data[self._curr_order[self._index_in_epoch:]]
            y_batch_feeded = self.label[self._curr_order[self._index_in_epoch:]]
            seq_length_feeded = self.seq_length[self._curr_order[self._index_in_epoch:]]

            if shuffle:
                np.random.shuffle(self._curr_order)

            X_batch_rest = self.data[self._curr_order[:num_rest]]
            y_batch_rest = self.label[self._curr_order[:num_rest]]
            seq_length_rest = self.seq_length[self._curr_order[:num_rest]]
            self._index_in_epoch = num_rest
            X_batch = np.concatenate((X_batch_feeded, X_batch_rest), axis=0)
            y_batch = np.concatenate((y_batch_feeded, y_batch_rest), axis=0)
            seq_length_batch = np.concatenate((seq_length_feeded, seq_length_rest), axis=0)
        else:
            X_batch = self.data[self._curr_order[self._index_in_epoch:self._index_in_epoch + batch_size]]
            y_batch = self.label[self._curr_order[self._index_in_epoch:self._index_in_epoch + batch_size]]
            seq_length_batch = self.seq_length[self._curr_order[self._index_in_epoch:self._index_in_epoch + batch_size]]
            self._index_in_epoch += batch_size
        return X_batch, y_batch, seq_length_batch

def get_rnn_cell(typ, platform, **kwargs):
    # get an rnn cell with the specified type on specific platform
    if typ == 'rnn' and platform == 'cpu':
        return tf.nn.rnn_cell.BasicRNNCell(**kwargs)
    elif typ == 'rnn' and platform == 'gpu':
        return tf.contrib.cudnn_rnn.CudnnRNNTanhSaveable(**kwargs)
    elif typ == 'lstm' and platform == 'cpu':
        return tf.nn.rnn_cell.LSTMCell(**kwargs)
    elif typ == 'lstm' and platform == 'gpu':
        return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(**kwargs)
    elif typ == 'gru' and platform == 'cpu':
        return tf.nn.rnn_cell.GRUCell(**kwargs)
    elif typ == 'gru' and platform == 'gpu':
        return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(**kwargs)
    else:
        print("Please choose a valid combination of type and platform")
        raise ValueError()

def get_review_data(filename, start, end):
    with open(filename) as f:
        data = f.readlines()
    reviews = [json.loads(x.strip()) for x in data]
    # print(reviews[0]['text'])
    sentences = [nltk.word_tokenize(reviews[i]['text'].lower()) for i in range(start, end)]

    return sentences

# return word2Vec model that can extract word embedding
def get_word_embedding(filename, start=0, end=1000):

    sentences = get_review_data(filename, start, end)
    saved_model = my_file = Path('model.txt')

    if not saved_model.is_file():
        
        model = Word2Vec(sentences, size=100, workers=8, sg=1, min_count=1)
        
        # save the trained model
        model.wv.save_word2vec_format('model.txt')
        saved_model = my_file = Path('model.txt')

    else:
        model = word2vec.KeyedVectors.load_word2vec_format('model.txt', binary=False)

    learned_vocab = list(model.wv.vocab)
    # print(model['pizza'])
    # print(list(learned_vocab))
    # print(model.most_similar(positive=[model['pizza']], topn=3))

    return model, sentences


def prepare_input_for_nn(model, sentences, n_steps=20, reverse=True, training=True):
    '''
    Prepare the input for the seq2seq model, with the pre-defined length and order.
    It is being said that reversed model leads to a better performance.
    n_steps = number of words we are going to feed into the network for the prediction of next.
    '''
    # list of numpy array (each is a embedding representing the previous sequence)
    inputs = []
    # list of vector (true word representing by vector)
    true_words = []
    # list of seq_lengths (the length of each sequence)
    seq_lengths=[]
    for sentence in sentences:
        sentence_embedding = []
        # get rid of reviews with smaller than 5 words
        if len(sentence) < 5:
            continue
        for i in range(len(sentence)):
            if sentence[i] in model.wv.vocab:
                sentence_embedding.append(model[sentence[i]])
            else:
                sentence_embedding.append(np.zeros(model.vector_size, dtype=np.float32))
        sentence_embedding = np.array(sentence_embedding)
        # begin prepare input and label
        for i in range(5, len(sentence)):
            seq_begin = np.max([0, i-n_steps])
            seq_length = np.min([i, n_steps])
            cur_input = sentence_embedding[seq_begin:i]
            cur_label = sentence[i]
            if seq_length < n_steps:
                pad_num = n_steps - seq_length
                pad = np.zeros((pad_num, model.vector_size))
                cur_input = np.concatenate((pad, cur_input), axis=0) 
            if reverse:
                cur_input=np.flip(cur_input,0)
            if cur_label in model.wv.vocab:
                inputs.append(cur_input)
                true_words.append(model[cur_label])
                seq_lengths.append(seq_length)
    inputs, true_words, seq_lengths = np.array(inputs, dtype=np.float32), np.array(true_words), np.array(seq_lengths)
    return DataSet(inputs, true_words, seq_lengths)


def build_nn(xpu, cell_type, training, input_ph, n_steps, n_inputs, n_neurons, seq_length_ph, out_size=100, keep_prob=0.5):
    cell = get_rnn_cell(typ=cell_type, platform=xpu, num_units = n_neurons)
    outputs, state = tf.nn.dynamic_rnn(cell, input_ph, dtype=tf.float32, sequence_length=seq_length_ph)
    output = tf.layers.dense(inputs=state[-1], units=out_size)
    output = tf.cond(training, lambda: tf.nn.dropout(output, keep_prob), lambda:output)
    output = tf.layers.dense(inputs=output, units=out_size)
    return output

def get_loss(pred_word, true_word):
    # consine distance
    loss = tf.losses.cosine_distance(tf.nn.l2_normalize(pred_word, 0), tf.nn.l2_normalize(true_word, 0), dim=0)
    return loss

def get_optimizer(loss, lr=0.005):
    # return a tf operation
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)



def train_nn(seq_length_ph,n_steps, n_inputs, training, model, sess, saver, input_ph, word_ph, loss, train_op, dataset, batch_size, num_epoch):
    print("begin training")

    for r in range(num_epoch):
        for i in range(dataset._num_data // batch_size + 1):
            X_batch, y_batch, seq_length_batch = dataset.next_batch(batch_size)
            #?????
            #print(X_batch.shape)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(train_op, feed_dict={training: True, input_ph: X_batch, word_ph: y_batch, seq_length_ph: seq_length_batch})
            cur_loss = sess.run(loss, feed_dict={training: True, input_ph: X_batch, word_ph: y_batch, seq_length_ph: seq_length_batch})
            if i%1000==0:
                print("loss for batch {} is {}".format(i, cur_loss))
        print("loss for epoch {} is {}".format(r, cur_loss))
    saver.save(sess, SAVE_PATH)


def get_prediction(seq_length_ph, training, model, nn_model, test_sentences, input_ph, word_ph, n_steps=20,reverse=True):
    print('begin predicting')
    dataset = prepare_input_for_nn(model, test_sentences, n_steps, reverse, training = False)
    X_batch, y_batch, seq_length_batch = dataset.data, dataset.label, dataset.seq_length
    test_inputs = X_batch
    test_true_words = y_batch
    print('test true word len = {}'.format(len(test_true_words)))
    #test_inputs = np.reshape(np.array(test_inputs), (len(test_inputs), model.vector_size))
    #test_true_words = np.reshape(np.array(test_true_words), (len(test_true_words), model.vector_size))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_PATH)
        test_pred_words = sess.run(nn_model, feed_dict={training: False, input_ph: test_inputs, word_ph: test_true_words, seq_length_ph: seq_length_batch})
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
    filename='yelp_academic_dataset_review.json'
    model, sentences = get_word_embedding(filename,0, 1000)
    #TODO: to change
    n_steps = 20
    reverse = True
    dataset = prepare_input_for_nn(model, sentences, n_steps, reverse)
    test_sentences = get_review_data(filename, 1000, 1200)

    print("----------------------- DONE WITH GET REVIEW DATA -----------------------")
    
    
    #embedded vector size
    n_inputs = model.vector_size
    #each vector is converted into dim=n_neurons
    n_neurons = 128
    batch_size=64
    # do reset_graph()?
    cpu_or_gpu = 'gpu'
    cell_ty = 'gru'
    
    input_ph = tf.placeholder(tf.float32, [None, n_steps, model.vector_size], name='train_input')
    word_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='train_label')
    training = tf.placeholder(tf.bool)
    seq_length_ph = tf.placeholder(tf.int32, [None])
    
    nn_model = build_nn(cpu_or_gpu, cell_ty, training, input_ph, n_steps, n_inputs, n_neurons, seq_length_ph)
    #state is the state of last time stamp (word) for EACH sentence

    loss = get_loss(nn_model, word_ph)
    train_op = get_optimizer(loss)
    saver = tf.train.Saver()
    # begin training
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        train_nn(seq_length_ph,n_steps, n_inputs, training, model, sess, saver, input_ph, word_ph, loss, train_op, dataset, batch_size, num_epoch=2)
    print("----------------------- DONE WITH TRAINING -----------------------")
    # t_input_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='test_input')
    # t_word_ph = tf.placeholder(tf.float32, [None, model.vector_size], name='test_predicted_label')
    test_true_words, test_pred_words = get_prediction(seq_length_ph, training, model, nn_model, test_sentences, input_ph, word_ph)
    print("----------------------- DONE WITH PREDICTION -----------------------")
    acc = get_accuracy(model, test_true_words, test_pred_words)
    print("----------------------- DONE WITH GET ACCURACY -----------------------")
    print('accuracy = {}'.format(acc))

if __name__ == '__main__':
    main()

