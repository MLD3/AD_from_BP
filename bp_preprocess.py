import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import copy

from file_locations import directories, data_file_name

'''
folder where data for preprocessing is contained
file_root should contain
    all_data.csv: 4 column table 
        first column - patient id 
        second column - number of days before alignment 
        third column - name of feature recorded
        fourth column - value of feature recorded
    labels.csv: 3 column table 
        first column - patient id 
        second column - patient age at alignment
        third column - whether patient had ad onset within 5 years of alignment
    pop.csv: 1 column table of patient ids

note: remove rows corresponding to column titles if present
'''
file_root =  directories['data_folder']

'''
reads file and turns to numpy array
'''
def get_file(file_name, dim):
    f = open(file_name, 'r')
    c = f.read()
    c = c[1:]
    c = c.replace('\n', ',')
    c = c.split(',')
    c = np.array(c)
    c = c[:-1]
    c = c.reshape((-1,dim))
    f.close()
    return c


'''
preprocess data
'''
def get_data():
    global file_root
    
    data_file = file_root + 'all_data.csv'
    label_file = file_root + 'labels.csv'
    pop_file = file_root + 'pop.csv'
    
    cov_data = get_file(data_file, 4)
    label_data = get_file(label_file, 4)
    pop = get_file(pop_file, 1)

    lab_keep = np.where(np.isin(label_data[:, 0], pop))[0]
    label_data = label_data[lab_keep, :]
    data_keep = np.where(np.isin(cov_data[:, 0], pop))[0]
    cov_data = cov_data[data_keep, :]
     
    return cov_data, label_data, pop.reshape(-1)


'''
get values within a range of time points
'''
def get_timeframe_vals(time_stamps, traj_vals, window_bounds):
    window_i = np.where(np.logical_and(time_stamps >= window_bounds[0], time_stamps < window_bounds[1]))[0]
    missing = 0
    if window_i.shape[0] == 0: #missing value
        missing = 1 #missing indicator
    window_stamps = time_stamps[window_i]
    window_vals = traj_vals[window_i]
    return missing, window_stamps, window_vals


'''
get evenly spaced trajectories for a specific numerical feature that changes over time (e.g., age, weight, bp)
'''
def get_trajectory(raw_data, pat, feature, window_size, val_assign):
    #isolate data for a specific patient (pat)
    pat_rows = np.where(raw_data[:, 0] == pat)[0]
    pat_data = raw_data[pat_rows, :]
    
    #find the relevant rows where the feature of interest (feature) is recorded, only keep rows with numerical values
    raw_traj_in = np.where(np.core.defchararray.find(pat_data[:, 2], feature) != -1)[0]
    raw_traj = pat_data[raw_traj_in, :]
    is_num = []
    for k in range(len(raw_traj)):
        try:
            float(raw_traj[k, 3])
            is_num.append(k)
        except ValueError:
            pass
    raw_traj = raw_traj[is_num, :]

    #find when the feature was recorded
    time_stamps_i = np.argsort(raw_traj[:, 1].astype(int))
    time_stamps = np.sort(raw_traj[:, 1].astype(int))
    traj_vals = raw_traj[time_stamps_i, 3].astype(float)

    if len(time_stamps) == 0:
        return np.array([[1, 0, 0]])

    #make empty, formatted array to fill in
    num_windows = int(np.ceil(np.max(raw_traj[:, 1][raw_traj[:, 1] != 'NULL'].astype(int)) / window_size))
    traj = np.zeros((num_windows, 3)) #indicator, value, number of measurements 
    if np.max(time_stamps) % window_size == 0:
        num_windows += 1
        traj = np.zeros((num_windows, 3))

    #fill in the formatted array
    curr_i, prev_i = 0, 0
    for i in range(num_windows):
        curr_stamp = time_stamps[curr_i]
        while curr_stamp < window_size*i and curr_i < len(time_stamps):
            curr_i += 1
            curr_stamp = time_stamps[curr_i]
            prev_i = curr_i - 1
        prev_stamp = time_stamps[prev_i]
        curr_meas = traj_vals[curr_i]
        prev_meas = traj_vals[prev_i]

        _, window_stamps, _ = get_timeframe_vals(time_stamps, traj_vals, [window_size*i, window_size*(i+1)])
        traj[i, 2] = np.unique(window_stamps).shape[0]

        if curr_stamp == window_size*i: 
            traj[i, 1] = curr_meas
        elif val_assign == 'carry_for':
            traj[i, 0] = 1
            traj[i, 1] = curr_meas
        elif val_assign == 'carry_back':
            traj[i, 0] = 1
            traj[i, 1] = prev_meas
        elif val_assign == 'impute':
            traj[i, 0] = 1
            traj[i, 1] = curr_meas
            if prev_stamp != curr_stamp:
                weight = (window_size*i - curr_stamp) / (prev_stamp - curr_stamp)
                traj[i, 1] = (1 - weight)*curr_meas + weight*prev_meas
    
    return traj


'''
get trajectories for all patients, over all features that need it
'''

def get_all_traj(feat, pop, window_size, val_assign):
    num_pat = pop.shape[0]
    traj_feats = ['age', 'BPSys', 'BPdia', 'weight'] #feature names as they appear in all_data.csv

    trajectories = {}
    for name in traj_feats: #record the minimum value, maximum value, and number of measurements for each patient
        trajectories[name] = []
        trajectories[name + '_min_val'] = []
        trajectories[name + '_max_val'] = []
        trajectories[name + '_num_meas'] = []

    for i in range(num_pat):
        if i % 500 == 0:
            print(i)
        pat = pop[i]
        for j in range(len(traj_feats)):
            traj = get_trajectory(feat, pat, traj_feats[j], window_size, val_assign)
            trajectories[traj_feats[j]].append(traj)
            had_val = np.where(traj[:, 1] > 0)[0] #not missing
            if had_val.shape[0] > 0:
                trajectories[traj_feats[j] + '_min_val'].append(np.min(traj[had_val, 1]))
                trajectories[traj_feats[j] + '_max_val'].append(np.max(traj[had_val, 1]))
                trajectories[traj_feats[j] + '_num_meas'].append(had_val.shape[0])

    for i in range(num_pat): #print out information on distribution of values observed
        if i % 500 == 0:
            print(i)
        pat = pop[i]
        for j in range(len(traj_feats)):
            print(np.percentile(trajectories[traj_feats[j] + '_min_val'], [0,1,20,40,50,60,80,99,100]))
            print(np.percentile(trajectories[traj_feats[j] + '_max_val'], [0,1,20,40,50,60,80,99,100]))
    
    #save data
    saved_data = open(file_root + 'processed_trajectories_' + val_assign + '_window' + str(window_size) + '.pkl', 'wb') 
    pickle.dump(trajectories, saved_data)
    saved_data.close()

    return trajectories


'''
exponent for variability independent of mean statistic
'''
def get_vim_exp(sds, means):
    vim_exp = np.zeros((sds.shape[1],))
    for i in range(sds.shape[1]):
        keep = np.intersect1d(np.where(sds[:, i] != 0)[0], np.where(means[:, i] != 0)[0])
        reg = LinearRegression().fit(np.log(means[keep, i]).reshape(-1, 1), np.log(sds[keep, i]))
        vim_exp[i] = reg.coef_[0]

    return vim_exp
    

'''
get summary statistics for trajectories
'''
def get_summ_stats(traj, pop):
    num_pat = pop.shape[0]

    num_feat = 6 #indicator, value 3 times
    avg, std = np.zeros((num_pat, num_feat)), np.zeros((num_pat, num_feat))
    val_range, arv = np.zeros((num_pat, num_feat)), np.zeros((num_pat, num_feat))
    vim, cvar = np.zeros((num_pat, num_feat)), np.zeros((num_pat, num_feat))
    traj_feats = ['BPSys', 'BPdia', 'weight'] #feature names as they appear in all_data.csv
    for i in range(num_pat):
        for j in range(len(traj_feats)):
            feat_name = traj_feats[j]   
            pat_traj = traj[feat_name][i]
            if pat_traj.shape[0] > 10:
                pat_traj = pat_traj[:10, :]
            if pat_traj.shape[0] == 0:
                avg[i, 2*j], std[i, 2*j], val_range[i, 2*j], arv[i, 2*j], vim[i, 2*j], cvar[i, 2*j] = 1, 1, 1, 1, 1, 1
                continue
            avg[i, 2*j+1], std[i, 2*j+1] = np.average(pat_traj[:, 1]), np.std(pat_traj[:, 1])
            val_range[i, 2*j+1] = np.max(pat_traj[:, 1]) - np.min(pat_traj[:, 1])
            if pat_traj.shape[0] < 2:
                arv[i, 2*j], std[i, 2*j], cvar[i, 2*j] = 1, 1, 1
            else:
                arv[i, 2*j+1] = np.average(np.absolute(pat_traj[:-1, 1] - pat_traj[1:, 1]))
                cvar[i, 2*j+1] = std[i, 2*j+1] / avg[i, 2*j+1]
       
    vim = np.zeros((num_pat, num_feat))
    vim_exp = get_vim_exp(std[:, [1, 3, 5]], avg[:, [1, 3, 5]])
    pop_means = np.average(avg, axis=0)
    for i in range(3):
        vim_denom = copy.deepcopy(avg[:, 2*i+1])
        vim_missing = np.where(vim_denom == 0)[0]
        vim[vim_missing, 2*i] = 1
        vim_denom[vim_denom == 0] = 1
        vim_denom = vim_denom ** vim_exp[i]
        vim[:, 2*i+1] = (std[:, 2*i+1] / vim_denom) * pop_means[2*i+1]**vim_exp[i]

    summ = {'avg': avg, 'std': std, 'avv': arv, 'range': val_range, 'vim': vim, 'cvar': cvar}

    return summ, vim_exp


'''
normalize to 0-1 range
'''
def normalize(val, bounds):
    if bounds[1] == bounds[0]:
        bounds[1] = bounds[1] + 1

    if (isinstance(val, int) or isinstance(val, float)) and val < bounds[0]:
        normalized = 0
    elif (isinstance(val, int) or isinstance(val, float)) and val > bounds[1]:
        normalized = 1
    else:
        normalized = (val - bounds[0]) / (bounds[1] - bounds[0])

    if not (isinstance(val, int) or isinstance(val, float)):
        normalized[normalized < 0] = 0
        normalized[normalized > 1] = 1

    return normalized


'''
format features as written below
features:
    0 - missing data for current window
    1 - age
    2 - sex
    3-7 - race (white, black, asian, american indian, other)
    8 - number of visits in time window
    9-11 - weight (indicator, value, num_measurements)
    12-14 - systolic bp (indicator, value, num_measurements) 
    15-17 - diastolic bp (indicator, value, num_measurements) 
    18-23 - avg over entire trajectory
    24-29 - std over entire trajectory
    30-35 - arv (average real variability) over entire trajectory
    36-41 - range over entire trajectory
    42-47 - vim over entire trajectory
    48-53 - cvar (coefficient of variation) over entire trajectory
'''
def format_features_even_spaced(raw_data, vis_days, trajectories, summ_stats, pat, vim_exp, window_size):
    num_feat = 54

    num_windows = int(np.ceil(np.max(vis_days) / window_size)) 
    if num_windows == 0:
        num_windows = 1
    vis_intervals = np.zeros((num_windows, 2))
    vis_intervals[:, 0] = window_size*np.arange(0, num_windows)
    vis_intervals[:, 1] = window_size*np.arange(1, num_windows+1)
    formatted_data = np.zeros((num_windows, num_feat))

    #sex
    s_i = np.where(np.core.defchararray.find(raw_data[:, 2], 'sex') != -1)[0]
    if len(s_i) > 0 and raw_data[s_i[0], 3] == 1: #1 for female, 0 for male
        formatted_data[:, 2] = 1
    #race
    r_i = np.where(np.core.defchararray.find(raw_data[:, 2], 'race') != -1)[0]
    if len(r_i) > 0 and raw_data[r_i[0], 3][0:] == 'C': #White (possible value for race feature that appears in all_data.csv)
        formatted_data[:, 3] = 1
    elif len(r_i) > 0 and raw_data[r_i[0], 3][0:] == 'AA': #Black (possible value for race feature that appears in all_data.csv)
        formatted_data[:, 4] = 1
    elif len(r_i) > 0 and raw_data[r_i[0], 3][0:] == 'A': #Asian ((possible value for race feature that appears in all_data.csv)
        formatted_data[:, 5] = 1
    elif len(r_i) > 0 and raw_data[r_i[0], 3][0:] == 'AI': #American Indian (possible value for race feature that appears in all_data.csv)
        formatted_data[:, 6] = 1
    else: #other race
        formatted_data[:, 7] = 1

    time_var = raw_data[np.where(raw_data[:, 1] != 'NULL')[0], :]
    for i in range(num_windows):
        day_l, day_u = vis_intervals[i, 0], vis_intervals[i, 1]
        vis_data = time_var[np.where(np.logical_and(time_var[:, 1].astype(int) >= day_l, time_var[:, 1].astype(int) < day_u))[0], :]
          
        #missing data
        if vis_data.shape[0] == 0:
            formatted_data[i, 0] = 1
            if i > 0:
                formatted_data[i, 1:] = formatted_data[i-1, 1:]
            continue
        #age
        a_i = np.where(vis_data[:, 2] == 'util-age')[0] #util-age is the feature name that appears in all_data.csv
        if len(a_i) > 0:
            age = np.average(vis_data[a_i, 3].astype(float))
            a_bounds = [np.percentile(trajectories['age_min_val'], 1), np.percentile(trajectories['age_max_val'], 99)]
            formatted_data[i, 1] = normalize(age, a_bounds)
        #number of visits
        num_vis = np.unique(vis_data[:, 1]).shape[0]
        nv_bounds = np.percentile(trajectories['age_num_meas'], [1, 99])
        num_vis = normalize(num_vis, nv_bounds)
        formatted_data[i, 8] = num_vis

    #weight and bp
    traj_feats = ['weight', 'BPSys', 'BPdia'] #feature names as they appear in all_data.csv
    for i in range(len(traj_feats)):
        feat_i = traj_feats[i]
        traj_i = trajectories[feat_i][pat][:num_windows, :]
        if traj_i.shape[0] == 0:
            traj_i = np.zeros((num_windows, 3))
            traj_i[:, 0] = 1
        formatted_data[:traj_i.shape[0], 9 + 3*i:12 + 3*i] = traj_i
        last_nonzero = np.where(formatted_data[:, 10 + 3*i] > 0)[0]
        if last_nonzero.shape[0] > 0 and last_nonzero[-1] < num_windows - 1:
            formatted_data[last_nonzero[-1] + 1:, 10 + 3*i] = formatted_data[last_nonzero[-1], 10 + 3*i]
        w_bounds = [np.percentile(trajectories[feat_i + '_min_val'], 1), np.percentile(trajectories[feat_i + '_max_val'], 99)]
        formatted_data[:, 10 + 3*i] = normalize(formatted_data[:, 10 + 3*i], w_bounds)
        w_bounds2 = np.percentile(trajectories[feat_i + '_num_meas'], [1, 99])
        formatted_data[:, 11 + 3*i] = normalize(formatted_data[:, 11 + 3*i], w_bounds2)
        last_nonzero = np.where(formatted_data[:, 10 + 3*i] > 0)[0]
        
    #summary stats
    stat_names = list(summ_stats.keys())
    for i in range(len(stat_names)):
        stat = stat_names[i]
        num_stat_feat = summ_stats[stat].shape[1]
        for j in range(num_stat_feat):
            val = summ_stats[stat][pat, j]
            if j % 2 == 1:
                have_stat = np.where(summ_stats[stat][:, j-1] == 0)[0]
                stat_bounds = np.percentile(summ_stats[stat][have_stat, j], [1, 99])
                val = normalize(val, stat_bounds)
            formatted_data[:, 18 + i*num_stat_feat + j] = val

    return formatted_data


'''
preprocess features
'''
def preprocess(feat, labs, pop, trajectories, summ_stats, vim_exp, window_size, val_assign):
    num_pat = pop.shape[0]
    max_lookback = 5000

    processed_feats = []
    seq_lens = []
    num_vis = []
    meas_density = []
    for i in range(num_pat): #process each patient individually
        if i % 500 == 0:
            print(i)
        pat = pop[i]
        pat_rows = np.where(feat[:, 0] == pat)[0]
        bp_vis = np.where(np.core.defchararray.find(feat[pat_rows, 2], 'BP') != -1)[0]
        bp_vis = np.intersect1d(bp_vis, np.where(feat[pat_rows, 3] != '')[0])
        vis_days = np.unique(feat[pat_rows, :][bp_vis, 1]).astype(int)
        vis_days = vis_days[vis_days <= max_lookback]
        pat_formatted = format_features_even_spaced(feat[pat_rows, :], vis_days, trajectories, summ_stats, i, vim_exp, window_size)
        meas_density.append(vis_days.shape[0])
        processed_feats.append(pat_formatted)
        seq_lens.append(pat_formatted.shape[0])
        num_vis.append(vis_days.shape[0])

    #package everyone's processed data into a dictionary
    processed_data = {\
        'features': processed_feats, \
        'lengths': seq_lens, \
        'num_vis': num_vis, \
        'labels': labs, \
        'population': pop, \
        'trajectories': trajectories, 
        'summ_stats': summ_stats, \
        'meas_density': meas_density, \
    }
    
    #save processed features
    saved_data = open(file_root + data_file_name, 'wb') 
    pickle.dump(processed_data, saved_data)
    saved_data.close()
    return processed_data


###################################################################################
'''
main block
'''
if __name__ == '__main__':
    np.random.seed(0)

    window_size = 183
    val_assign = 'carry_for' #there are also options for 'carry back', 'average', and 'impute'

    print('preparing trajectories')
    feat, lab, pop = get_data()

    get_all_traj(feat, pop, window_size, val_assign)
    trajectories = open(file_root + 'processed_trajectories_' + val_assign + '_window' + str(window_size) + '.pkl', 'rb')
    traj = pickle.load(trajectories)
    trajectories.close()

    print('preprocessing')
    summ_stats, vim_exp = get_summ_stats(traj, pop)
    preprocess(feat, lab, pop, traj, summ_stats, vim_exp, window_size, val_assign)
    
