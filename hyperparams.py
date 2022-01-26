'''
record of hyperparameters 
'''

###################################################################################################
'''
only the value at 'layer_s' matters here since we are loading a pre-trained model and the layer size needs to match
ranges for tunining are not included here because we are loading a model that has already been trained
'''

no_bp_hyperparams = \
{
    'baseline1': {'l_rate': 0.0001, 'l2': 0.0001, 'batch': 5, 'n_layer': 3, 'layer_s': 75, 'long': 10000, 'weight': [1, 65]}, \
} 

bp_stats_hyperparams = \
{
    'baseline1': {'l_rate': 0.0001, 'l2': 0.001, 'batch': 5, 'n_layer': 3, 'layer_s': 75, 'long': 10000, 'weight': [1, 65]}, \

}

bp_traj_hyperparams = \
{
    'baseline1': {'l_rate': 0.0001, 'l2': 0.0001, 'batch': 5, 'n_layer': 3, 'layer_s': 75, 'long': 10000, 'weight': [1, 65]}, \
}

###################################################################################################
'''
putting everything together
'''
all_hyperparams = \
{
    'no_bp': no_bp_hyperparams, \
    'bp_stats': bp_stats_hyperparams, \
    'bp_traj': bp_traj_hyperparams, \
}

###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
