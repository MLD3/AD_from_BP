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
    c = c[1:] #c[1:] works on linux, c[3:] works on windows
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
def get_rdw(file_name):
    dataset = pickle.load(open(file_name, 'rb')) 
    feats = dataset['features']
    lens = np.array(dataset['lengths'])
    labs = dataset['labels']
    pop = np.array(dataset['population'])
    num_meas = np.array(dataset['meas_density'])
    
    #put labels in order
    ordered_labs = np.zeros(labs.shape, dtype=object)
    for i in range(pop.shape[0]):
        pat = pop[i]
        pat_i = np.where(labs[:, 0] == pat)[0]
        ordered_labs[i, :] = labs[pat_i, :]

    #want to only look at patients with >=35 measurements, coment out this part if using all patients
    len_valid = np.where(num_meas >= 35)[0]
    feats = [feats[i] for i in len_valid]
    lens = lens[len_valid]
    ordered_labs = ordered_labs[len_valid]
    pop = pop[len_valid]
    num_meas = num_meas[len_valid]

    return feats, lens, ordered_labs[:, 2].astype(int), pop, num_meas, dataset['trajectories']


###################################################################
'''
filter data
'''
def filter_data(raw_data, lengths, labs, time_series, full_stats):
    if not time_series and not full_stats: #no bp
        for i in range(len(raw_data)):
            raw_data[i] = raw_data[i][:, :12]

    elif not time_series and full_stats: #bp stats
        for i in range(len(raw_data)):
            stat_vals = np.arange(18, 54, 1)
            raw_data[i] = np.concatenate((raw_data[i][:, :12], raw_data[i][:, stat_vals]), axis=1) 
            raw_data[i] = np.delete(raw_data[i], np.array([22,23,28,29,34,35,40,41,46,47,52,53]).astype(int) - 6, axis=1)

    elif time_series and not full_stats: #bp traj
        for i in range(len(raw_data)):
            raw_data[i] = raw_data[i][:, :18] 

    return raw_data


'''
split data into training, validation, and test sets
'''
def split_data(raw_data, lengths, labs, bp, traj):
    #filter out length 0 and normalize data
    raw_data = filter_data(raw_data, lengths, labs, bp, traj)
    len_valid = np.where(lengths > 0)[0]
    raw_data = [raw_data[i] for i in len_valid]
    lengths = lengths[len_valid]
    labs = labs[len_valid]
    print('prop convert: ', np.sum(labs) / labs.shape[0], len(raw_data))
    
    max_len = 11 #do not use time steps 12, 13, ...
    lengths[lengths > max_len] = max_len
    print('num_features', raw_data[0].shape[1])
    
    whole_package = [[torch.Tensor(raw_data[i]) for i in range(len(raw_data))], lengths, labs]
    
    return [whole_package]


'''
get and preprocess dataset by name
'''
def get_dataset(name, params=[]):
    dataset = 0
    if name == 'no_bp':
        dataset = get_rdw(features)
        return split_data(dataset[0], dataset[1], dataset[2], False, False)
    elif name == 'bp_stats':
        dataset = get_rdw(features)
        return split_data(dataset[0], dataset[1], dataset[2], False, True)
    elif name == 'bp_traj':
        dataset = get_rdw(features)
        return split_data(dataset[0], dataset[1], dataset[2], True, False)


###################################################################
'''
main block
'''
if __name__ == '__main__':
    print(':)')
