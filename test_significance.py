import numpy as np
import torch
import random
import pickle
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

import matplotlib
matplotlib.rcParams.update({'font.size': 16})

'''
where evaluations are stored
'''
from file_locations import directories, data_file_name, result_ext

results_dir = directories['results_folder']


###################################################################################################
'''
get from file
'''
def get_res(setting):
    results = pickle.load(open(results_dir + setting, 'rb'))

    return results


'''
do a resampling test
'''
def test_sig(setting1, setting2):
    res1 = get_res(setting1)
    res2 = get_res(setting2)

    num_metrics = 2
    sig_vals = {}
    for i in range(num_metrics):
        metric = ['auroc', 'aupr'][i]
        res1_i = res1[metric]
        res2_i = res2[metric]
        nboot = res1_i.shape[0]
        sig_vals[metric] = 2*min(sum(np.subtract(res1_i,res2_i)>0)/nboot, sum(np.subtract(res1_i,res2_i)<0)/nboot)

    return sig_vals


'''
plot roc and pr curves
'''
def plot_curves(settings, setting_names):
    x_ax = np.linspace(0, 1, 100)
    curves = ['roc', 'pr']
    metric_names = ['AUROC', 'AUPR']
    colors = ['C2', 'C4', 'C5'] 
    legend_locs = ['lower right', 'upper right']
    xlabs = ['1 - Specificity', 'Sensitivity']
    ylabs = ['Sensitivity', 'Precision']

    plt.figure(figsize=(12, 5.5))

    for i in range(len(settings)):
        setting = settings[i]
        res_i = get_res(setting)
        disc_i = [res_i['auroc'], res_i['aupr']]
        curves_i = [res_i['curves']['roc'], res_i['curves']['pr']]
        for j in range(len(curves)):
            median = np.median(curves_i[j], axis=0)
            lower = np.percentile(curves_i[j], 2.5, axis=0)
            upper = np.percentile(curves_i[j], 97.5, axis=0)
            disc_med = np.median(disc_i[j])
            disc_lower = np.percentile(disc_i[j], 2.5)   
            disc_upper = np.percentile(disc_i[j], 97.5)
            plot_lab = setting_names[i] + '\n' + '%.2f' % disc_med + ' (' + '%.2f' % disc_lower + ', ' + '%.2f' % disc_upper + ')'
            plt.subplot(1, 2, j+1)
            plt.plot(x_ax, median, label=plot_lab, color=colors[i])
            plt.fill_between(x_ax, lower, upper, alpha=0.2, color=colors[i])
            plt.legend(loc=legend_locs[j])
            plt.xlabel(xlabs[j])
            plt.ylabel(ylabs[j])
            plt.title(metric_names[j])
            plt.xlim(left=0, right=1)
            plt.ylim(bottom=0, top=1)
            if j == 1:
                plt.xlim(left=0, right=0.5)
    plt.suptitle('Michigan Medicine (MM)')
    plt.savefig('curves.png')    

    plt.show()


'''
statistical significance of demographics between populations
contingency table - 
    rows are groups (va dev, va val, rdw val)
    columns are banner points/cuts (what you're looking at in each group, in this case patient sex)
    rdw means mm here
'''
def test_pop_sig():
    cont_tab = np.array([[5488-153, 153], [1372-27, 27]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for prop female va dev/va val:', test_res[1])

    cont_tab = np.array([[5488-153, 153], [1201-666, 666]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for prop female va dev/rdw val:', test_res[1])

    cont_tab = np.array([[1201-666, 666], [1372-27, 27]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for prop female va val/rdw val:', test_res[1])

    ###########################
    cont_tab = np.array([[133-4, 4], [33-2, 2]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for prop female va dev/va val among ads:', test_res[1])

    cont_tab = np.array([[133-4, 4], [30-20, 20]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for prop female va dev/rdw val among ads:', test_res[1])

    cont_tab = np.array([[33-2, 2], [30-20, 20]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for prop female va val/rdw val among ads:', test_res[1])

    ###########################
    cont_tab = np.array([[5488-2873, 2873], [1372-718, 718]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for bpdia >= 78 va dev/va val:', test_res[1])

    cont_tab = np.array([[5488-2873, 2873], [1201-257, 257]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for bpdia >= 78 va dev/rdw val:', test_res[1])

    cont_tab = np.array([[1201-257, 257], [1372-718, 718]])
    test_res = chi2_contingency(cont_tab)
    print('stat sig for bpdia >= 78 va val/rdw val:', test_res[1])


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    datasets = ['no_bp', 'bp_stats', 'bp_traj']
    for i in range(len(datasets)):
        for j in range(i+1, len(datasets)):
            res1 = datasets[i] + result_ext
            res2 = datasets[j] + result_ext
            print(datasets[i] + ' vs ' + datasets[j] + ' significance: ', test_sig(res1, res2))

    names = ['No BP', 'BP Stats', 'BP Trajectories']
    res = [datasets[i] + result_ext for i in range(len(datasets))]
    plot_curves([res[0], res[1], res[2]], names)

    test_pop_sig()

