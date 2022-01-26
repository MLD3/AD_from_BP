import numpy as np
import torch
import random

import train_get_data as get_data
import train_util as util
from train_hyperparams import all_hyperparams, hp_ranges
from train_data_settings import all_settings 


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    #random seed
    random.seed(0)
    np.random.seed(0) 
    torch.manual_seed(0) 

    dataset_name = 'bp_traj' 
    approach = 'baseline1'
    tune = True
    boot = True
    
    data_params = all_settings[dataset_name]
    data_package = get_data.get_dataset(dataset_name, data_params)
    
    hyperparams = all_hyperparams[dataset_name][approach]
    hyperparam_ranges = hp_ranges[dataset_name][approach]
    
    data_params['n_feats'] = data_package[0][0][0].shape[1]
    
    if tune:
        mod, hyperparams, results = util.tune_hyperparams(data_package, approach, data_params, hyperparam_ranges, boot)
    else:
        mod, _, results = util.get_model(data_package, approach, data_params, hyperparams, boot)
    
    print(dataset_name, approach, hyperparams)
    print(hyperparam_ranges)
    print(results)


