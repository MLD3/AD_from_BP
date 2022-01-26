import numpy as np
import torch
import random

import get_data
import util
from hyperparams import all_hyperparams
from data_settings import all_settings 


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    #random seed
    random.seed(0)
    np.random.seed(0) 
    torch.manual_seed(0) 
    
    dataset_name = 'bp_traj' #options: no_bp, bp_stats, bp_traj
    approach = 'baseline1' #standard lstm
    
    data_params = all_settings[dataset_name]
    data_package = get_data.get_dataset(dataset_name, data_params)
    
    hyperparams = all_hyperparams[dataset_name][approach]
    
    data_params['n_feats'] = data_package[0][0][0].shape[1]
    mod, results = util.get_model(data_package, approach, data_params, hyperparams)
    if dataset_name == 'bp_traj':
        util.analyze_mod(approach, hyperparams, data_params, data_package)
    

