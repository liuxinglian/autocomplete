# model2_config.py

'''
parameters specific to model 2
'''

class model2_params:
    def __init__(self):
        self.train_size = 100
        self.test_size = (int)(0.2 * self.train_size)
        self.train_start = 0
        self.train_end = self.train_start + self.train_size
        self.test_start = self.train_end
        self.test_end = self.test_start + self.test_size
        self.tf_save_path = './model2_train_{}_test_{}/m.cpkt'.format(self.train_size, self.test_size)
        self.add_star = False
        self.epoches = 3
        self.learning_rate = 0.001