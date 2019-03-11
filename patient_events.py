import pandas as pd
import numpy as np
from math import ceil

'''
The baselines are: Ethnicity, Gender, Age, Height, Weight
'''

data_path = '/data/suparna/MGP_data/'

def prep_baseline_mgp():
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    #subject_ids = [int(x[0]) for x in lines]
    subject_ids = [20, 30, 33, 52, 85, 123, 124, 135, 2492, 2488]
    #the episode1 gives baseline and mortality label
    #the episode 1 timeseries gives the timed events to be fed to rnn
    #the input events CV has both the subjects' data
    end_times = []
    num_obs_times = []
    num_obs_values = []
    num_rnn_grid_times = []
    rnn_grid_times = []
    labels = []
    T = []
    Y = []
    baseline_covs = []
    meds_on_grid = []
    ind_kf = []
    ind_kt = []

    Ethnicity = {}
    eth_count = 0
    Gender = {'M':0, 'F':1}

    for sub in enumerate(subject_ids):
        Y_i = []
        ind_kf_i = []
        ind_kt_i = []
        stays = pd.read_csv(data_path+'/root/'+str(sub)+'/stays.csv', parse_dates=True)
        #timeline = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/episode1_timeseries.csv')
        intime = pd.to_datetime(stays['INTIME'])
        label = stays['MORTALITY_INHOSPITAL'][0]
        for stay_no in range(stays.shape[0]):
            timeline = pd.read_csv(data_path+'/root/'+str(sub)+'/episode'+str(stay_no)+'.csv')
            timeline = timeline[timeline.Hours>=0]
            T_i = timeline.Hours
            column_map = {n:i for i,n in enumerate(list(timeline.columns))}
            len_columns = len(timeline.columns)
            for t in range(T_i.shape[0]):
                values = timeline.iloc[t]
                presence = timeline.iloc[t].isnull()
                for i in range(1,len_columns):
                    if presence[i]==False:
                        if type(values[i]) is not str:
                            Y_i.append(values[i])
                            ind_kf_i.append(i-1)
                            ind_kt_i.append(t)
            end_times.append(timeline.iloc[-1][0])
            num_obs_times.append(timeline.count()[0])
            #num_obs_values.append(np.sum(timeline.count()[1:]))
            num_obs_values.append(len(Y_i))
            num_rnn_grid_times.append(ceil(end_times[-1])+1)
            rnn_grid_times.append(list(np.arange(num_rnn_grid_times[-1])))
            baseline = pd.read_csv(data_path+'/root/'+str(sub)+'/baseline'+str(stay_no)+'.csv')
            baseline_i = baseline.iloc[0]
            medicines = pd.read_csv(data_path+'/medicines/'+str(sub)+'_stay'+str(stay_no)+'.med')
            meds_on_grid_i = medicines.to_numpy()
            meds_on_grid.append(meds_on_grid_i)
            baseline_covs.append(baseline_i)
            Y.append(Y_i)
            ind_kf.append(ind_kf_i)
            ind_kt.append(ind_kt_i)
            T.append(T_i.tolist())
            labels.append(label)
    print num_obs_times
    print num_obs_values
    print num_rnn_grid_times
    print rnn_grid_times
    print labels
    print T
    print Y
    print ind_kf
    print ind_kt
    print meds_on_grid
    print baseline_covs
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

if __name__=="__main__":
    dataset_prep()
