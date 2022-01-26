import pickle

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import torch

from file_locations import directories, data_file_name

data_dir = directories['data_folder']
features = data_dir + data_file_name

'''
reads file and turns to numpy array
'''
def get_file(file_name, dim):
    f = open(file_name, 'r')
    c = f.read()
    c = c[1:] #use 1 as the starting index for linux, 3 for windows
    c = c.replace('\n', ',')
    c = c.split(',')
    c = np.array(c)
    c = c[:-1]
    c = c.reshape((-1,dim))
    f.close()
    return c

'''
get from file, see bp_preprocess.py for feature definitions
'''
def get_va(file_name):
    dataset = pickle.load(open(file_name, 'rb'))
    feats = dataset['features']
    lens = np.array(dataset['lengths'])
    labs = dataset['labels']
    pop = dataset['population']

    #put labels in order
    ordered_labs = np.zeros(labs.shape, dtype=object)
    for i in range(pop.shape[0]):
        pat = pop[i]
        pat_i = np.where(labs[:, 0] == pat)[0]
        ordered_labs[i, :] = labs[pat_i, :]

    return feats, lens, ordered_labs[:, 2].astype(int)


###################################################################
'''
make a label for each length/outcome (e.g., outcome 1 with length 2 gets its own label)
'''
def get_length_labs(lengths, labs):
    max_length = np.max(lengths)
    new_labs = np.zeros(labs.shape)
    num_outcomes = np.unique(labs).shape[0]
    
    for i in range(num_outcomes):
        new_labs[labs == i] = (i * max_length) + lengths[labs == i]
    
    return labs


'''
filter data
'''
def filter_data(raw_data, lengths, labs, time_series, full_stats): 
    if not time_series and not full_stats: #no bp
        for i in range(len(raw_data)):
            raw_data[i] = raw_data[i][:, :12]

    elif not time_series and full_stats: #bp stats
        del_cols = [22,23,28,29,34,35,40,41,46,47,52,53]
        for i in range(len(raw_data)):
            stat_vals = np.arange(18, 54, 1) 
            raw_data[i] = np.concatenate((raw_data[i][:, :12], raw_data[i][:, stat_vals]), axis=1) 
            raw_data[i] = np.delete(raw_data[i], np.array(del_cols).astype(int)-6, axis=1)

    elif time_series and not full_stats: #bp traj
        for i in range(len(raw_data)):
            raw_data[i] = raw_data[i][:, :18] 

    return raw_data


'''
split data into training, validation, and test sets
the inputs bp and traj are boolean values for whether to include bp and trajectory stats information into the data
'''
def split_data(raw_data, lengths, labs, bp, traj):
    raw_data = filter_data(raw_data, lengths, labs, bp, traj)
    len_valid = np.where(lengths > 0)[0]
    raw_data = [raw_data[i] for i in len_valid]
    lengths = lengths[len_valid]
    labs = labs[len_valid]
    print('prop convert: ', np.sum(labs) / labs.shape[0])
    
    #stratify by length and label
    max_len = 11
    lengths[lengths > max_len] = max_len
    raw_data = [raw_data[i][:lengths[i], :] for i in range(len(lengths))]
    print('num feats: ', raw_data[0].shape[1])
    length_labs = get_length_labs(lengths, labs)
    
    #split into training/not training
    dummy = np.zeros((len(raw_data), 2))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
    train_i, test_i = next(splitter.split(dummy, length_labs))

    train_data = [torch.Tensor(raw_data[i]) for i in train_i]
    train_lengths = lengths[train_i]
    train_labs = labs[train_i]
    
    pretest_data = [torch.Tensor(raw_data[i]) for i in test_i]
    pretest_lengths = lengths[test_i]
    pretest_labs = labs[test_i]
    
    #further split test set into test/validation
    dummy = np.zeros((len(pretest_data), 2))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    not_train = test_i
    test_i, val_i = next(splitter.split(dummy, length_labs[test_i]))
    test_data = [pretest_data[i] for i in test_i]
    test_lengths = pretest_lengths[test_i]
    test_labs = pretest_labs[test_i]
    
    val_data = [pretest_data[i] for i in val_i]
    val_lengths = pretest_lengths[val_i]
    val_labs = pretest_labs[val_i]
    
    #package for convenience
    train_package = [train_data, train_lengths, train_labs]
    test_package = [test_data, test_lengths, test_labs]
    validation_package = [val_data, val_lengths, val_labs]
    whole_package = [[torch.Tensor(raw_data[i]) for i in range(len(raw_data))], lengths, labs]
    
    return train_package, test_package, validation_package, whole_package


'''
get and preprocess dataset by name
'''
def get_dataset(name, params=[]):
    dataset = 0
    if name == 'no_bp':
        dataset = get_va()
        return split_data(dataset[0], dataset[1], dataset[2], False, False)
    elif name == 'bp_stats':
        dataset = get_va()
        return split_data(dataset[0], dataset[1], dataset[2], False, True)
    elif name == 'bp_traj':
        dataset = get_va()
        return split_data(dataset[0], dataset[1], dataset[2], True, False)


###################################################################
'''
main block
'''
if __name__ == '__main__':
    print(':)')
