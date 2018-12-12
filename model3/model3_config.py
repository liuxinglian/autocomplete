# model3_config.py

'''
parameters specific to model 3
'''

class model3_params:
    def __init__(self):
        self.add_star = False
        self.epoches = 3
        self.learning_rate = 0.001
        self.is_shuffle = True
        
        self.num_steps = 50
        self.reverse = True

        self.num_neurons = 128
        self.batch_size = 64
        self.cpu_or_gpu = 'gpu'
        self.cell_type = 'gru'
        self.num_layers = 3
        self.if_bidirect = True

        self.train_size = 100
        self.train_start = 0
        self.train_end = self.train_start + self.train_size

        self.test_size = (int)(0.2 * self.train_size)
        self.test_start = self.train_end
        self.test_end = self.test_start + self.test_size
        self.tf_save_path = './model3_train_{}_test_{}/m.cpkt'.format(self.train_size, self.test_size)
