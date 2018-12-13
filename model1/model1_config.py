# model1_config.py

'''
parameters specific to model 1
'''

class model1_params:
    def __init__(self):
        self.n = 2 
        self.topn = 10

        self.add_star = False
        self.is_shuffle = True
        
        self.train_size = 10000
        self.train_start = 0
        self.train_end = self.train_start + self.train_size
        
        self.test_size = (int)(0.2 * self.train_size)
        self.test_start = self.train_end
        self.test_end = self.test_start + self.test_size
        
        self.save_path = './model1_train_{}_test_{}_n_{}/save.p'.format(self.train_size, self.test_size, self.n)
