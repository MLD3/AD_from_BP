import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

from scipy.stats import loguniform
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold

import torch
import torch.optim as optim
import torch.nn as nn

import baseline1 #standard lstm
from file_locations import directories, model_names

mod_dir = directories['mod_folder']


################################################################################################
'''
set up a model for training based on the approach and hyperparameters
'''
def initialize_model(approach, hyperparameters, data_params):
    model, loss_fx = 1, 1
    
    if approach == 'baseline1':
        model = baseline1.baseline1_net(hyperparameters, data_params)
        loss_fx = baseline1.baseline1_loss(hyperparameters, data_params)
    
    return model, loss_fx


'''
train a model, evaluate performance every 10 epochs
'''
def train_model(model, loss_fx, hyperparams, train_data, val_data, data_params):
    #unpack
    train_cov, train_lens, train_labs = train_data[0], train_data[1], train_data[2]
    val_cov, val_lens, val_labs = val_data[0], val_data[1], val_data[2]
    train_dummy = np.zeros((len(train_cov), 1))
    
    #setup
    l_rate, l2_const, num_batch = hyperparams['l_rate'], hyperparams['l2'], hyperparams['batch']
    mod_params = model.get_parameters()
    optimizer = optim.Adam(mod_params, lr=l_rate, weight_decay=l2_const) 
    min_epochs = 11
    max_epochs = 200
    
    #initial evaluation, val_loss refers to evaluation on the validation set, not the value of the loss function
    vsort_i = np.flip(np.argsort(val_lens))
    val_out = model([val_cov[i] for i in vsort_i], val_lens[vsort_i], True)
    val_loss = -evaluate(model, val_data)['auroc']
    loss_diff = copy.deepcopy(val_loss)
    loss_prev = copy.deepcopy(val_loss)
    loss_tol = 1e-4
    
    #train model 
    i = 0
    prev_mod, prev_loss = copy.deepcopy(model), 0
    while (loss_diff > loss_tol or i < min_epochs) and i < max_epochs:
        train_loss = 0     
        splitter = KFold(num_batch, shuffle=True) 
        for _, batch_ind in splitter.split(train_dummy):
            sort_i = np.flip(np.argsort(train_lens[batch_ind]))
            b_cov, b_len, b_lab = [train_cov[j] for j in batch_ind], train_lens[batch_ind], train_labs[batch_ind]
            train_out = model([b_cov[j] for j in sort_i], b_len[sort_i], True)
            batch_loss = loss_fx(train_out, torch.Tensor(b_lab[sort_i]))
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += (batch_loss.detach() / num_batch)
   
        #evaluate every 10 epochs on validation data (not the held out test set)
        if i % 10 == 0 and i > 0:
            val_out = model([val_cov[j] for j in vsort_i], val_lens[vsort_i], True)
            val_loss = -evaluate(model, val_data)['auroc']
            loss_diff = loss_prev - val_loss
            loss_prev = copy.deepcopy(val_loss)
            if loss_diff > 0:
                prev_mod = copy.deepcopy(model)
                prev_loss = val_loss
            
            val_eval = evaluate(model, val_data)
            train_eval = evaluate(model, train_data)
            print('new training evaluation')
            print(i, val_loss, loss_diff)
            print(val_eval)
            print('training loss: ', train_loss)
            print(train_eval)
            
        i += 1
    
    print('done training')  
    return prev_mod, prev_loss


################################################################################################
'''
measure model performance
'''
def evaluate(model, eval_data, get_curves=False):
    cov, lens, labs = eval_data[0], eval_data[1], eval_data[2]
    results = {}
    
    sort_i = np.flip(np.argsort(lens))
    preds = model.eval().forward([cov[i] for i in sort_i], lens[sort_i]).detach().numpy()
    results['auroc'] = roc_auc_score(labs[sort_i], preds[:, 1])
    results['aupr'] = average_precision_score(labs[sort_i], preds[:, 1])
    
    model.train()
    
    if get_curves:
        roc = roc_curve(labs[sort_i], preds[:, 1])
        pr = precision_recall_curve(labs[sort_i], preds[:, 1])
        return results, roc, pr
    
    return results


################################################################################################
'''
overall wrapper - train/test/validate a model given the dataset, approach, parameters
'''
def get_model(dataset_package, approach, data_params, hyperparams):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    
    model, loss_fx = initialize_model(approach, hyperparams, data_params)
    
    val_res = 0
    model, val_res = train_model(model, loss_fx, hyperparams, train_data, val_data, data_params)
    
    test_results = evaluate(model, test_data)

    #torch.save(model.state_dict(), mod_dir + model_names[data_params['name']])
    
    return model, val_res, test_results


'''
hyperparameter tuning
'''
def tune_hyperparams(dataset_package, approach, data_params, hyperparam_ranges, boot=True):
    budget = 10
    keys = list(hyperparam_ranges.keys())
    num_hyperparams = len(keys)
    
    test_data = dataset_package[1]
    val_results = 1000 * np.ones((budget,)) 
    
    best_hyperparams = 1
    best_mod = 1
    
    for i in range(budget):
        hyperparams = {}
        for j in range(num_hyperparams):
            bound = hyperparam_ranges[keys[j]]
            if bound[0] < bound[1] and keys[j] not in ['n_layer', 'layer_s']:
                hyperparams[keys[j]] = loguniform.rvs(bound[0], bound[1])
            elif bound[0] < bound[1] and keys[j] == 'n_layer':
                hyperparams[keys[j]] = np.random.randint(bound[0], bound[1])
            elif bound[0] < bound[1] and keys[j] == 'layer_s':
                hyperparams[keys[j]] = np.random.randint(bound[0], bound[1]) * 25
            else:
                hyperparams[keys[j]] = bound[0]
       
        print(i, hyperparams) 
        mod, val_res, _ = get_model(dataset_package, approach, data_params, hyperparams)
        val_results[i] = val_res
        
        if val_res == np.min(val_results): #want lowest value because storing negative aurocs
            best_mod = mod
            best_hyperparams = hyperparams
    if boot:
        print('bootstrapping')
        print(bootstrap_results(best_mod, test_data, data_params['name']))
        print('done bootstrapping')

    test_results = evaluate(best_mod, test_data)
    torch.save(best_mod.state_dict(), mod_dir + mod_dir + model_names[data_params['name']])

    return best_mod, best_hyperparams, test_results


################################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
