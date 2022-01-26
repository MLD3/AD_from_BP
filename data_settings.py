'''
record of all settings for datasets, only the value at 'name' matters here
''' 

no_bp = \
{
    'name': 'no_bp', \
    'n_class': 2, \
    'n_feats': 1, \
    'prop_pos': 0.02, \
    'weights': [1, 75], \
    'val_weights': 1, \
    'diff_weights': 1, \
} 

bp_stats = \
{
    'name': 'bp_stats', \
    'n_class': 2, \
    'n_feats': 1, \
    'prop_pos': 0.02, \
    'weights': [1, 75], \
    'val_weights': 1, \
    'diff_weights': 1, \
} 

bp_traj = \
{
    'name': 'bp_traj', \
    'n_class': 2, \
    'n_feats': 1, \
    'prop_pos': 0.02, \
    'weights': [1, 75], \
    'val_weights': 1, \
    'diff_weights': 1, \
} 
##########################################################################################################
all_settings = \
{
    'no_bp': no_bp, \
    'bp_stats': bp_stats, \
    'bp_traj': bp_traj, \
}

##########################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
