Instructions for use:

Fill in the directory names in file_locations.py
    data_folder: where all of the data for preprocessing is stored, see bp_preprocess.py line 10 for more detail
    mod_folder: where all of the models will be saved
    results_folder: where the results will be saved

Fill in (in file_locations.py)
    data_file_name: which is where the preprocessed data will be stored after running bp_preprocess.py
    result_ext: which is a suffix/tag that will be added to the names of result files (this should end in '.pkl')

Collect the data specified in bp_preprocess.py, line 10

Run bp_preprocess.py to preprocess the data
    The names on lines 151,212,298,302,304,306,308,325,337 may need to be changed depending on how the variables in all_data are named

For training:
    Fill in the empty strings for the model names in the model_names dictionary in file_locations.py
    Run train_run.py 3 times
        First time: set dataset_name (line 21) to no_bp 
        Second time: set dataset_name (line 21) to bp_stats 
        Third time: set dataset_name (line 21) to bp_traj 

For validation:
    Set data_file_name in file_locations.py to where the validation data is stored
    Set result_ext in file_locations.py to any suffix/tag desired (should end in '.pkl')
    Check the following names and change if needed:
        util.py, line 108
        check_stats.py, line 73

    Run check_stats.py - this will give cohort characteristics
    Run run.py 3 times 
        First time: set dataset_name (line 21) to no_bp [for bootstrapped results]
        Second time: set dataset_name (line 21) to bp_stats [for bootstrapped results]
        Third time: set dataset_name (line 21) to bp_traj [for bootstrapped results and plots of median trajectories]
    Run test_significance.py - this will give the auroc and aupr curves as well as p values for significance tests
