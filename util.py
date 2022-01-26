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

import baseline1 #a standard lstm
from file_locations import directories, data_file_name, result_ext, model_names

mod_dir = directories['mod_folder']
results_dir = directories['results_folder']
data_dir = directories['data_folder']

data_file = data_dir + data_file_name
nobp_mod = mod_dir + model_names['no_bp']
stats_mod = mod_dir + model_names['bp_stats']
traj_mod = mod_dir + model_names['bp_traj'] 


################################################################################################
'''
set up a model for training based on the approach and hyperparameters
'''
def initialize_model(approach, hyperparameters, data_params):
    model = baseline1.baseline1_net(hyperparameters, data_params)
    loss_fx = baseline1.baseline1_loss(hyperparameters, data_params)
    
    return model, loss_fx


################################################################################################
'''
measure auroc and aupr
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


'''
plot median trajectories for low and high risk groups
'''
def examine_trajectories(model, cov, lens, labs, hyperparams, data_params, val_data):
    sort_i = np.flip(np.argsort(lens))
    preds = model.forward([cov[i] for i in sort_i], lens[sort_i], return_extra=True, get_grad=True)

    val_sort_i = np.flip(np.argsort(val_data[1]))
    val_preds = model.forward([val_data[0][i] for i in val_sort_i], val_data[1][val_sort_i], return_extra=True, get_grad=True)

    plt.figure(figsize=(12, 6))

    #get risk groups
    len_thresh = 11
    long_traj = np.where(lens[sort_i] >= len_thresh)[0]
    thresh_preds = preds['preds'][long_traj, lens[long_traj] - 1, :][:, 1].detach().numpy()
    num_risk_groups = 10 #deciles to easily separate the top and bottom 10%
    pos_preds = preds['preds'][long_traj, lens[long_traj] - 1, :][:, 1].detach().numpy()
    sorted_preds = np.argsort(pos_preds)
    groups = []
    for i in range(num_risk_groups): #manually get percentiles
        in_group = sorted_preds[i*int(long_traj.shape[0]/num_risk_groups):(i+1)*int(long_traj.shape[0]/num_risk_groups)]
        if i == num_risk_groups - 1:
            in_group = sorted_preds[i*int(long_traj.shape[0]/num_risk_groups):]
        groups.append(in_group)

    #format input for plotting
    inp = preds['inp_for_plots'].detach().numpy()
    evals = []
    for group in groups: #groups correspond to quantiles
        group_eval = np.zeros((len_thresh, inp.shape[2], group.shape[0]))
        for i in range(group.shape[0]):
            group_eval[:, :, i] += inp[long_traj[group[i]], :len_thresh, :]
        last_zero = np.where(group_eval[-1, -1, :] == 0)[0]
        evals.append(group_eval)

    dataset = pickle.load(open(data_file, 'rb'))
    trajectories = dataset['trajectories']

    plot_evals = [0, -1] #choosing which quantiles to evaluate (0 and -1 correspond to the bottom and top 10%)
    num_rows, num_cols = 2, len(plot_evals) // 2
    feat_labs = ['BPSys', 'BPdia'] #features as they appear in all_data.csv
    feat_title = ['SBP (mmHg)', 'DBP (mmHg)']
    feat_offset = 13 #13 for bpsys

    #unnormalize the values to get the raw BP measurements
    norm_offsets, norm_mults, norm_bounds, meas_bounds = [], [], [], []
    for i in range(len(feat_labs)):
        feat_i = feat_labs[i]
        bounds = [np.percentile(trajectories[feat_i + '_min_val'], 1), np.percentile(trajectories[feat_i + '_max_val'], 99)]
        if feat_offset == 14:
            bounds = [np.percentile(trajectories[feat_i + '_num_meas'], 1), np.percentile(trajectories[feat_i + '_num_meas'], 99)]
        offset = bounds[0]
        mult = bounds[1] - bounds[0]
        norm_offsets.append(offset)
        norm_mults.append(mult)
        norm_bounds.append(bounds)
        meas_bounds.append([np.percentile(trajectories[feat_i + '_num_meas'], 1), np.percentile(trajectories[feat_i + '_num_meas'], 99)]) 
    norm_bounds.append([np.percentile(trajectories['age_min_val'], 1), np.percentile(trajectories['age_max_val'], 99)])
    meas_bounds.append([np.percentile(trajectories['age_num_meas'], 1), np.percentile(trajectories['age_num_meas'], 99)])

    #plot the median trajectories
    risk_lab = ['Low risk', 'High risk']
    num_rows, num_cols = 1, len(feat_labs)
    for i in range(len(feat_labs)):
        feat_i = feat_offset+3*i
        plt.subplot(num_rows, num_cols, i+1)
        for j in range(2):
            eval_j = evals[plot_evals[j]]
            unnorm_vals = (eval_j[:, feat_i, :] * norm_mults[i]) + norm_offsets[i] 
            meds = np.median(unnorm_vals, axis=1)
            error_bars = np.zeros((2, len_thresh))
            error_bars[1, :] = meds - np.percentile(unnorm_vals, 25, axis=1)
            error_bars[0, :] = np.percentile(unnorm_vals, 75, axis=1) - meds
            plt.errorbar(np.flip(np.arange(len_thresh)), meds, yerr=np.flip(error_bars, axis=0), \
                         label=risk_lab[j], alpha=0.9, marker='o')
            plt.xticks(np.arange(len_thresh), np.flip(np.arange(len_thresh)*6))
            plt.xlabel('Months Prior\nto Alignment')
            plt.legend(loc='upper left')
            plt.ylabel(feat_title[i])
    plt.suptitle('Michigan Medicine (MM)')
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(results_dir + 'median_trajectories' + result_ext[:-4] + '.png')
    return 1


'''
analyze what model does
'''
def analyze_mod(approach, hyperparams, data_params, dataset):
    model, _ = initialize_model(approach, hyperparams, data_params)
    model.load_state_dict(torch.load(traj_mod)) 
    
    test_data = dataset[0]
    cov, lens, labs = test_data[0], test_data[1], test_data[2]
    sort_i = np.flip(np.argsort(lens))
    print('test set size', lens.shape)

    disc_perf = evaluate(model, test_data)
    print('discriminative performance: ', disc_perf)

    #get median trajectories
    grad_res = examine_trajectories(model, cov, lens, labs, hyperparams, data_params, test_data)
    
    return 1


################################################################################################
'''
bootstrap results
'''
def bootstrap_results(model, eval_data, dataset, num_boots=1000):
    evals = {}
    cov, lens, labs = eval_data[0], eval_data[1], eval_data[2]
    num_data = len(cov)
    pos = np.where(labs==1)[0]
    neg = np.where(labs==0)[0]

    prop_pos = np.zeros((num_boots,))

    x_ax = np.linspace(0, 1, 100)
    all_tp = {'roc': [], 'pr': []}
    
    for i in range(num_boots):
        if i % 100 == 0:
            print('bootstrap iteration ', i)
        boot_i = np.random.choice(num_data, size=(num_data,))
        boot_data = [[cov[j] for j in boot_i], lens[boot_i], labs[boot_i]]
        prop_pos[i] = np.sum(boot_data[2]) / boot_data[2].shape[0]
        results, roc, pr = evaluate(model, boot_data, get_curves=True)    
        for metric in results:
            if i == 0:
                evals[metric] = np.zeros((num_boots,))
            evals[metric][i] = results[metric]
        all_tp['roc'].append(np.interp(x_ax, roc[0], roc[1]).reshape(1, -1))
        all_tp['pr'].append(np.interp(x_ax, pr[0], pr[1]).reshape(1, -1))

    all_tp['roc'] = np.concatenate(all_tp['roc'], axis=0)
    all_tp['pr'] = np.concatenate(all_tp['pr'], axis=0)
    evals['curves'] = all_tp

    saved_evals = open(results_dir + dataset + result_ext, 'wb') #results from va model, number of encounters adjusted
    pickle.dump(evals, saved_evals)
    saved_evals.close()  

    print('auorc bootstrap, 95% CI: ', np.percentile(evals['auroc'], [2.5, 50, 97.5]))
    print('aupr bootstrap, 95% CI: ', np.percentile(evals['aupr'], [2.5, 50, 97.5]))
    
    return evals


################################################################################################
'''
overall wrapper - train/test/validate a model given the dataset, approach, parameters
'''
def get_model(dataset_package, approach, data_params, hyperparams):
    test_data = dataset_package[0]
    
    model, loss_fx = initialize_model(approach, hyperparams, data_params)
    if data_params['name'] == 'no_bp':
        model.load_state_dict(torch.load(nobp_mod))  
    elif data_params['name'] == 'bp_stats':
        model.load_state_dict(torch.load(stats_mod))
    elif data_params['name'] == 'bp_traj':
        model.load_state_dict(torch.load(traj_mod))
    
    test_results = evaluate(model, test_data)
    bootstrap_results(model, test_data, data_params['name'])
    
    return model, test_results


################################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
