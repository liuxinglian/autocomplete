from pathlib import Path
from operator import itemgetter
import json
import numpy as np
import nltk
import tensorflow as tf
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
from dict_filter import get_esaved
from model3_config import model3_params
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system_config import system_params
from prep_data import get_review_data, get_word_embedding
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

class DataSet(object):
    '''
    This is an object that is holding the data and generate batches.
    '''
    def __init__(self, 
                 data,
                 label,
                 seq_length,
                 stars,
                 training=True):
        self.data = data
        self.label = label
        self.seq_length = seq_length
        self.training = training
        self.stars = stars
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_data = data.shape[0]
        self._curr_order = np.arange(self._num_data)

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        #print(self.training)
        if self.training == False:
            return self.data, self.label, self.seq_length, self.stars
        if start == 0 and self._epochs_completed == 0 and shuffle:
            np.random.shuffle(self._curr_order)
        if start + batch_size > self._num_data:
            self._epochs_completed += 1
            num_feeded = self._num_data - start
            num_rest = batch_size - num_feeded
            X_batch_feeded = self.data[self._curr_order[self._index_in_epoch:]]
            y_batch_feeded = self.label[self._curr_order[self._index_in_epoch:]]
            seq_length_feeded = self.seq_length[self._curr_order[self._index_in_epoch:]]
            stars_feeded = self.stars[self._curr_order[self._index_in_epoch:]]

            if shuffle:
                np.random.shuffle(self._curr_order)

            X_batch_rest = self.data[self._curr_order[:num_rest]]
            y_batch_rest = self.label[self._curr_order[:num_rest]]
            seq_length_rest = self.seq_length[self._curr_order[:num_rest]]
            stars_rest = self.seq_length[self._curr_order[:num_rest]]
            self._index_in_epoch = num_rest
            X_batch = np.concatenate((X_batch_feeded, X_batch_rest), axis=0)
            y_batch = np.concatenate((y_batch_feeded, y_batch_rest), axis=0)
            seq_length_batch = np.concatenate((seq_length_feeded, seq_length_rest), axis=0)
            stars_batch = np.concatenate((stars_feeded, stars_rest), axis=0)
        else:
            X_batch = self.data[self._curr_order[self._index_in_epoch:self._index_in_epoch + batch_size]]
            y_batch = self.label[self._curr_order[self._index_in_epoch:self._index_in_epoch + batch_size]]
            seq_length_batch = self.seq_length[self._curr_order[self._index_in_epoch:self._index_in_epoch + batch_size]]
            stars_batch = self.stars[self._curr_order[self._index_in_epoch:self._index_in_epoch + batch_size]]
            self._index_in_epoch += batch_size
        return X_batch, y_batch, seq_length_batch, stars_batch


def get_rnn_cell(att, typ, platform, **kwargs):
    # get an rnn cell with the specified type on specific platform
    if typ == 'rnn' and platform == 'cpu':
        cell= tf.nn.rnn_cell.BasicRNNCell(**kwargs)
    #elif typ == 'rnn' and platform == 'gpu':
        #return tf.contrib.cudnn_rnn.CudnnRNNTanhSaveable(**kwargs)
    elif typ == 'lstm' and platform == 'cpu':
        cell= tf.nn.rnn_cell.LSTMCell(**kwargs)
    elif typ == 'lstm' and platform == 'gpu':
        cell= tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(**kwargs)
    #elif typ == 'lstmbn' and (platform == 'cpu' or platform == 'gpu'):
        #return tf.contrib.rnn.LayerNormBasicLSTMCell(**kwargs)
    elif typ == 'gru' and platform == 'cpu':
        cell= tf.nn.rnn_cell.GRUCell(**kwargs)
    elif typ == 'gru' and platform == 'gpu':
        cell= tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(**kwargs)
    else:
        print("Please choose a valid combination of type and platform")
        raise ValueError()
    
    if att:
        cell=tf.contrib.rnn.AttentionCellWrapper(cell, 3)
    return cell


def prepare_input_for_nn(wv_model, sentences, n_steps, stars, reverse=True, training=True):
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
    seq_lengths = []
    # list of stars (the stars of each review)
    stars_list = []

    for k in range(len(sentences)):
        sentence = sentences[k]
        star = stars[k]
        sentence_embedding = []
        # get rid of reviews with smaller than 5 words
        if len(sentence) < 5:
            continue
        for i in range(len(sentence)):
            if sentence[i] in wv_model.wv.vocab:
                sentence_embedding.append(wv_model[sentence[i]])
            else:
                sentence_embedding.append(np.zeros(wv_model.vector_size, dtype=np.float32))
        sentence_embedding = np.array(sentence_embedding)
        # begin prepare input and label
        for i in range(5, len(sentence)):
            seq_begin = np.max([0, i-n_steps])
            seq_length = np.min([i, n_steps])
            cur_input = sentence_embedding[seq_begin:i]
            cur_label = sentence[i]
            if seq_length < n_steps:
                pad_num = n_steps - seq_length
                pad = np.zeros((pad_num, wv_model.vector_size))
                cur_input = np.concatenate((pad, cur_input), axis=0) 
            if reverse:
                cur_input = np.flip(cur_input,0)
            if cur_label in wv_model.wv.vocab:
                inputs.append(cur_input)
                true_words.append(wv_model[cur_label])
                seq_lengths.append(seq_length)
                stars_list.append(star)
    inputs, true_words, seq_lengths = np.array(inputs, dtype=np.float32), np.array(true_words), np.array(seq_lengths)
    stars_list = np.array(stars_list, dtype=np.int)
    return DataSet(inputs, true_words, seq_lengths, stars_list)


def build_nn(n_layers, xpu, cell_type, training, stars, input_ph, n_steps, num_inputs, n_neurons, seq_length_ph, out_size=100, keep_prob=0.5, bidirection=False, attention = False):
    
    if n_layers ==1:
        print("1 layer")
        cell = get_rnn_cell(att=attention, typ=cell_type, platform=xpu, num_units = n_neurons, )
    else:
        print("many layers")
        stacked_cells = [get_rnn_cell(att=attention, typ=cell_type, platform=xpu, num_units = n_neurons) for _ in range(n_layers)]
        cell =tf.contrib.rnn.MultiRNNCell(stacked_cells)
    if not bidirection:
        print("forward")
        outputs, state = tf.nn.dynamic_rnn(cell, input_ph, dtype=tf.float32, sequence_length=seq_length_ph)
        print(tf.shape(outputs), tf.shape(state))
    else:
        print("bidirection")
        cell_bw = get_rnn_cell(att=attention, typ=cell_type, platform=xpu, num_units = n_neurons)
        outputs_fb, state_fb = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, input_ph, dtype=tf.float32, sequence_length=seq_length_ph)
        #state_fb=[<tf.Tensor 'bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 128) dtype=float32>, <tf.Tensor 'bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 128) dtype=float32>]
        #state_fw h : outputs[-1][:,0:dim,:](last t)
        #bw h: first t
        #last time
        state = state_fb[-1]
        #first time
        #state = state_fb[0]

    if n_layers !=1:
        state = state[-1]
    if cell_type=='lstm' or cell_type == "lstmbn":
        print("LSTM")
        state = state[-1]
        
    #reshape stars from ? to ?,1
    stars=tf.reshape(stars,[-1,1])
    #add starts in
    state = tf.concat((state, stars), 1)
    output = tf.layers.dense(inputs=state, units=out_size) 
    output = tf.cond(training, lambda: tf.nn.dropout(output, keep_prob), lambda:output)
    output = tf.layers.dense(inputs=output, units=out_size)
    return output


def get_loss(pred_word, true_word):
    # consine distance
    loss = tf.losses.cosine_distance(tf.nn.l2_normalize(pred_word, 0), tf.nn.l2_normalize(true_word, 0), dim=0)
    return loss


def get_optimizer(loss, lr=0.003):
    # return a tf operation
    return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99).minimize(loss)


def train_nn(seq_length_ph,n_steps, num_inputs, training, wv_model, sess, saver,stars_ph, input_ph, word_ph, loss, train_op, dataset, batch_size, num_epoch):
    print("begin training")

    for r in range(num_epoch):
        for i in range(dataset._num_data // batch_size + 1):
            X_batch, y_batch, seq_length_batch, stars_batch = dataset.next_batch(batch_size)
            #?????
            #print(X_batch.shape)
            X_batch = X_batch.reshape((-1, n_steps, num_inputs))
            sess.run(train_op, feed_dict={training: True, stars_ph:stars_batch, input_ph: X_batch, word_ph: y_batch, seq_length_ph: seq_length_batch})
            cur_loss = sess.run(loss, feed_dict={training: True, stars_ph:stars_batch,input_ph: X_batch, word_ph: y_batch, seq_length_ph: seq_length_batch})
            if i%1000==0:
                print("loss for batch {} is {}".format(i, cur_loss))
        print("loss for epoch {} is {}".format(r, cur_loss))
    saver.save(sess, SAVE_PATH)


def get_prediction(seq_length_ph, training, wv_model, nn_model, test_sentences, stars, stars_ph, input_ph, word_ph, n_steps,reverse=True):
    print('begin predicting')
    dataset = prepare_input_for_nn(wv_model, test_sentences, n_steps, stars, reverse, training = False)
    X_batch, y_batch, seq_length_batch = dataset.data, dataset.label, dataset.seq_length
    test_stars = dataset.stars
    test_inputs = X_batch
    test_true_words = y_batch
    print('test true word len = {}'.format(len(test_true_words)))
    #test_inputs = np.reshape(np.array(test_inputs), (len(test_inputs), wv_model.vector_size))
    #test_true_words = np.reshape(np.array(test_true_words), (len(test_true_words), wv_model.vector_size))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_PATH)
        test_pred_words = sess.run(nn_model, feed_dict={training: False, stars_ph:test_stars, input_ph: test_inputs, word_ph: test_true_words, seq_length_ph: seq_length_batch})
    print('test pred word len = {}'.format(len(test_pred_words)))
    return test_true_words, test_pred_words


def get_accuracy(wv_model, true_words, pred_words, topn=10):
    print('begin getting accuracy')
    correct = 0
    for i in range(len(true_words)):

        true_word_vec = true_words[i]
        pred_word_vec = pred_words[i]

        temp1 = wv_model.most_similar([true_word_vec], [], topn)
        true_words_set = set([pair[0] for pair in temp1])
        temp2 = wv_model.most_similar([pred_word_vec], [], topn)
        pred_words_set = set([pair[0] for pair in temp2])

        if bool(true_words_set & pred_words_set):
            correct += 1
    return correct / len(true_words)


def main():
    sys_params = system_params()
    model_params = model3_params()

    save_path = model_params.tf_save_path
    save_folder = os.path.dirname(save_path)
    while os.path.isdir(save_folder):
        overwrite = input("There is a existing model on this path, overwrite? [y/n]")
        if (overwrite == 'y'):
            shutil.rmtree(save_folder)

    start_train, end_train = model_params.train_start, model_params.train_end
    start_test, end_test = model_params.test_start, model_params.test_end
    
    #filename = sys_params.all_reviews_jsonfn
    filename = 'partial_reviews1000.json'
    start_train = 0
    end_train = 100
    start_test = 101
    end_test = 120
    wv_model, sentences, stars = get_word_embedding(filename, start_train, end_train)
    test_sentences, stars = get_review_data(filename, start_test, end_test, model_params.is_shuffle)
    dataset = prepare_input_for_nn(wv_model, sentences, model_params.num_steps, stars,  model_params.reverse)

    print("----------------------- DONE WITH GET REVIEW DATA -----------------------")
    
    #embedded vector size
    num_inputs = wv_model.vector_size
    #each vector is converted into dim=n_neurons
    
    # do reset_graph()?
    
    input_ph = tf.placeholder(tf.float32, [None, model_params.num_steps, num_inputs], name='train_input')
    stars_ph = tf.placeholder(tf.float32, [None], name='train_star_input')
    word_ph = tf.placeholder(tf.float32, [None, num_inputs], name='train_label')
    training = tf.placeholder(tf.bool)
    seq_length_ph = tf.placeholder(tf.int32, [None])
    
    nn_model = build_nn(
        model_params.num_layers, 
        model_params.cpu_or_gpu, 
        model_params.cell_type, 
        training, 
        stars_ph, 
        input_ph, 
        model_params.num_steps, 
        num_inputs, 
        model_params.num_neurons, 
        seq_length_ph, 
        bidirection=model_params.if_bidirect)
    #state is the state of last time stamp (word) for EACH sentence

    loss = get_loss(nn_model, word_ph)
    train_op = get_optimizer(
        loss,
        lr=model_params.learning_rate)
    saver = tf.train.Saver()
    # begin training
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        train_nn(seq_length_ph,n_steps, num_inputs, training, wv_model, sess, saver, stars_ph,input_ph, word_ph, loss, train_op, dataset, batch_size, num_epoch=3)
    print("----------------------- DONE WITH TRAINING -----------------------")
    # t_input_ph = tf.placeholder(tf.float32, [None, wv_model.vector_size], name='test_input')
    # t_word_ph = tf.placeholder(tf.float32, [None, wv_model.vector_size], name='test_predicted_label')
    test_true_words, test_pred_words = get_prediction(seq_length_ph, training, wv_model, nn_model, test_sentences,  stars, stars_ph, input_ph, word_ph, n_steps)
    print("----------------------- DONE WITH PREDICTION -----------------------")
    #acc = get_accuracy(wv_model, test_true_words, test_pred_words)
    #print("----------------------- DONE WITH GET ACCURACY -----------------------")
    #print('accuracy = {}'.format(acc))
    eSaved = get_esaved(wv_model, test_true_words, test_pred_words, topn=1, cons=20)
    print("----------------------- DONE WITH GET ESAVED -----------------------")
    print('eSaved = {}'.format(eSaved))

if __name__ == '__main__':
    main()
