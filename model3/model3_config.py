# model3_config.py

'''
parameters specific to model 2
'''

model3_params = {
    train_size = 10000,
    test_size = (int)(0.2 * train_size),
    
    train_start = 0,
    train_end = train_start + train_size,

    test_start = train_end,
    test_end = test_start + test_size,

    tf_save_path : './model' + '_train_' + train_size + '_test_' + test_size + '/m.cpkt',

    add_star : False
}