'''
record of all hyper parameters 
'''

###################################################################################################
'''
specific hyperparameter settings 
'''

no_bp_hyperparams = \
{
     'baseline1': {'l_rate': 0.002, 'l2': 1e-7, 'batch': 5, 'n_layer': 3, 'layer_s': 75, \
                 'long': 10000, 'weight': [1, 40]}, \
} 
    
bp_stats_hyperparams = \
{
    'baseline1': {'l_rate': 0.0015, 'l2': 1e-7, 'batch': 5, 'n_layer': 3, 'layer_s': 80, \
                 'long': 10000, 'weight': [1, 42]}, \
} 

bp_traj_hyperparams = \
{
    'baseline1': {'l_rate': 0.0019, 'l2': 1e-7, 'batch': 5, 'n_layer': 3, 'layer_s': 75, \
                 'long': 10000, 'weight': [1, 40]}, \
} 


'''
ranges for tuning
'''
va_ranges = \
{
    'baseline1': {'l_rate': [1e-6, 1e-2], 'l2': [1e-7, 1e-2], 'batch': [5, 5], 'n_layer': [1, 4], 'layer_s': [1, 4], \
                   'long': [10000, 10000], 'weight':[(1, 40), (1, 40)]}, \
}
    
va_stats_ranges = \
{
    'baseline1': {'l_rate': [1e-6, 1e-2], 'l2': [1e-7, 1e-2], 'batch': [5, 5], 'n_layer': [1, 4], 'layer_s': [1, 4], \
                   'long': [10000, 10000], 'weight':[(1, 40), (1, 40)]}, \
}
    
va_traj_ranges = \
{
    'baseline1': {'l_rate': [1e-6, 1e-2], 'l2': [1e-7, 1e-2], 'batch': [5, 5], 'n_layer': [1, 4], 'layer_s': [1, 4], \
                   'long': [10000, 10000], 'weight':[(1, 40), (1, 40)]},
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

hp_ranges = \
{
    'no_bp': no_bp_ranges, \
    'bp_stats': no_bp_stats_ranges, \
    'bp_traj': bp_traj_ranges, \
}

###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
