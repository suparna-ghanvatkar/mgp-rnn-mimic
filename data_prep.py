import pandas as pd
import numpy as np
from math import ceil

def dataset_prep():
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

    medseries = pd.read_csv('/data/MIMIC3/INPUTEVENTS_CV.csv', low_memory=False)
    nmeds = medseries.ITEMID.unique()
    med_map = {n:i for i,n in enumerate(nmeds)}

    for sub in subject_ids:
        Y_i = []
        ind_kf_i = []
        ind_kt_i = []
        stays = pd.read_csv('~/mimic3-benchmarks/data/root/train/'+str(sub)+'/stays.csv', parse_dates=True)
        timeline = pd.read_csv('~/mimic3-benchmarks/data/root/train/'+str(sub)+'/episode1_timeseries.csv')
        intime = pd.to_datetime(stays['INTIME'])
        label = stays['MORTALITY_INHOSPITAL'][0]
        timeline = timeline[timeline.Hours>=0]
        T_i = timeline.Hours
        #column_map = {n:i for i,n in enumerate(list(timeline.columns))}
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
        num_obs_values.append(np.sum(timeline.count()[1:]))
        num_rnn_grid_times.append(ceil(end_times[-1])+1)
        rnn_grid_times.append(list(np.arange(num_rnn_grid_times[-1])))
        baseline_i = []
        if stays.ETHNICITY[0] not in Ethnicity.keys():
            Ethnicity[stays.ETHNICITY[0]] = eth_count
            eth_count+=1
        baseline_i.append(Ethnicity[stays.ETHNICITY[0]])
        baseline_i.append(Gender[stays.GENDER[0]])
        baseline_i.append(stays.AGE[0])
        meds_i = []
        med = medseries[medseries.SUBJECT_ID==sub]
        medtimes = pd.to_datetime(med.CHARTTIME)
        medt = medtimes.apply(lambda x: x-intime)[0]
        medt = medt.dt.total_seconds()/3600
        med.CHARTTIME = medt
        med = med[med.CHARTTIME>=0]
        med = med.sort_values('CHARTTIME')
        medtimes = med.CHARTTIME.unique()
        med_i_t = [0]*len(nmeds)
        prev_t = 0
        for m in med.itertuples():
            m = m._asdict()
            if m['CHARTTIME']==prev_t:
                med_i_t[med_map[m['ITEMID']]]=m['AMOUNT']
            else:
                meds_i.append(med_i_t)
                med_i_t = [0]*len(nmeds)
                med_i_t[med_map[m['ITEMID']]]=m['AMOUNT']
        meds_i.append(med_i_t)
        meds_on_grid_i = []
        start_i = 0
        end_i = 1
        rnn_t = 0
        #print sub, len(medtimes), num_rnn_grid_times[-1]
        for t in medtimes:
            #print t
            if t<=rnn_grid_times[-1][rnn_t]:
                end_i+=1
            else:
                m_t = meds_i[start_i:end_i]
                m = [sum(x) for x in zip(*m_t)]
                meds_on_grid_i.append(m)
                start_i = end_i
                rnn_t+=1
                end_i+=1
        while rnn_t<num_rnn_grid_times[-1]:
            meds_on_grid_i.append([0]*len(nmeds))
            rnn_t+=1
        meds_on_grid.append(meds_on_grid_i)
        baseline_covs.append(baseline_i)
        Y.append(Y_i)
        ind_kf.append(ind_kf_i)
        ind_kt.append(ind_kt_i)
        T.append(T_i.tolist())
        labels.append(label)
    #print num_obs_times
    #print num_obs_values
    #print num_rnn_grid_times
    #print rnn_grid_times
    #print labels
    #print T
    #print Y
    #print ind_kf
    #print ind_kt
    #print meds_on_grid
    #print baseline_covs
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

if __name__=="__main__":
    dataset_prep()
