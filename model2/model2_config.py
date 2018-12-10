# model2_config.py

'''
parameters specific to model 2
'''

class model2_params:
    def __init__(self):
        self.train_size = 10000
        self.test_size = (int)(0.2 * train_size)
        self.train_start = 0
        self.train_end = train_start + train_size
        self.test_start = train_end
        self.test_end = test_start + test_size
        self.tf_save_path = './model2' + '_train_' + train_size + '_test_' + test_size + '/m.cpkt'
        self.add_star = False
        self.epoches = 3