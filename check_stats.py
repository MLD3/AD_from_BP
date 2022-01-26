import numpy as np
import get_data

from file_locations import directories, data_file_name

data_dir = directories['data_folder']

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

###################################################################################################
def get_conv(labels):
    num_conv = np.sum(labels)
    prop_cov = num_conv / labels.shape[0]
    print('AD converters', num_conv, prop_cov*100)
    return 1

def get_num_meas(num_meas):
    print('Number of measurements (med, iqr)', np.percentile(num_meas, [50, 25, 75]))
    return 1

def get_amnt_long(lengths):
    print('Amount of longitudinal data in years (med, iqr)', np.percentile(lengths, [50, 25, 75]) / 2)
    return 1

def get_sex(feats):
    num_pats = len(feats)
    fcount = 0
    for i in range(num_pats):
        if feats[i][0, 2] == 1:
            fcount += 1
    print('Number female', fcount, fcount*100 / num_pats)
    print('Number male', num_pats - fcount, 100 - (fcount / num_pats)*100)
    return 1

def get_race(feats):
    num_pats = len(feats)
    rcounts = np.zeros((5,))
    for i in range(num_pats):
        rcounts += feats[i][0, 3:8]
    print('Race counts (White, Black, Asian, American Indian, Other)', rcounts, rcounts*100 / num_pats)
    return 1

def get_comorbids(pop):
    num_pats = pop.shape[0]
    #dyslipidemia, dl_list is an array of patient ids for patients who have dyslipidemia
    dl_list = get_data.get_file(data_dir + 'dyslipidemia.csv', 1)
    num_dl = np.intersect1d(pop, dl_list).shape[0]
    print('Dyslipidemia', num_dl, num_dl / num_pats)
    #kidney disease, kd_list is an array of patient ids for patients who have kidney disease
    kd_list = get_data.get_file(data_dir + 'kidney_disease.csv', 1)
    num_kd = np.intersect1d(pop, kd_list).shape[0]
    print('Kidney disease', num_kd, num_kd / num_pats)
    #diabetes, db_list is an array of patient ids for patients who have diabetes
    db_list = get_data.get_file(data_dir + 'diabetes.csv', 1)
    num_db = np.intersect1d(pop, db_list).shape[0]
    print('Diabetes', num_db, num_db / num_pats)
    return 1

def get_bp(trajectories, feats):
    feat_labs = ['BPSys', 'BPdia']
    offsets = []
    mults = []
    for i in range(len(feat_labs)):
        feat_i = feat_labs[i]
        bounds = [np.percentile(trajectories[feat_i + '_min_val'], 1), np.percentile(trajectories[feat_i + '_max_val'], 99)]
        offset = bounds[0]
        mult = bounds[1] - bounds[0]
        offsets.append(offset)
        mults.append(mult)

    num_data = len(feats)
    bpsys = np.zeros((num_data,))
    bpdia = np.zeros((num_data,))
    for i in range(num_data):
        bpsys[i] = (feats[i][0, 13] * mults[0]) + offsets[0]
        bpdia[i] = (feats[i][0, 16] * mults[1]) + offsets[1]
    print('bpsys values (med, iqr)', np.percentile(bpsys, [50, 25, 75]))
    print('bpdia values (med, iqr)', np.percentile(bpdia, [50, 25, 75]))
    print('number at least 78: ', np.where(bpdia >= 78)[0].shape) 


def get_missing_bp(trajectories, feats):
    missing_dbp = np.zeros((11,))
    missing_sbp = np.zeros((11,))
    total = np.zeros((11,))

    #indexes 14, 17 show number of measurements for sbp and dbp, respectively
    for i in range(len(feats)):
        pat_feats = feats[i]
        for j in range(11):
            if j+1 > pat_feats.shape[0]:
                continue
            total[j] += 1
            if pat_feats[j, 14] == 0:
                missing_sbp[j] += 1
            if pat_feats[j, 17] == 0:
                missing_dbp[j] += 1
   
    print('percent missing by timepoint (alignment going back)')
    print('sbp', missing_sbp / total)
    print('dbp', missing_dbp / total)


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    feats, lens, labs, pop, num_meas, trajectories = get_data.get_rdw(get_data.features)
    
    keep = np.arange(lens.shape[0])
    keep = np.where(labs == 1)[0]
    feats = [feats[keep[i]] for i in range(len(keep))]
    lens = lens[keep]
    labs = labs[keep]
    pop = pop[keep]
    num_meas = num_meas[keep]

    get_conv(labs)
    get_num_meas(num_meas)
    get_amnt_long(lens)
    get_sex(feats)
    get_race(feats)
    get_comorbids(pop)
    get_bp(trajectories, feats)
    get_missing_bp(trajectories, feats)

    print(':)')

