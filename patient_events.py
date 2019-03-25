import pandas as pd
import numpy as np
from math import ceil, isnan
import pickle
from collections import defaultdict

'''
The baselines are: Ethnicity, Gender, Age, Height, Weight
'''

data_path = '/data/suparna/MGP_data/'

def retrieve_high_mimic_dataset():
    num_obs_times = pickle.load(open('num_obs_times_high_mimic.pickle','r'))
    num_obs_values = pickle.load(open('num_obs_values_high_mimic.pickle','r'))
    num_rnn_grid_times = pickle.load(open('num_rnn_grid_times_high_mimic.pickle','r'))
    rnn_grid_times = pickle.load(open('rnn_grid_times_high_mimic.pickle','r'))
    labels = pickle.load(open('labels_high_mimic.pickle','r'))
    T = pickle.load(open('T_high_mimic.pickle','r'))
    Y = pickle.load(open('Y_high_mimic.pickle','r'))
    ind_kf = pickle.load(open('ind_kf_high_mimic.pickle','r'))
    ind_kt = pickle.load(open('ind_kt_high_mimic.pickle','r'))
    meds_on_grid = pickle.load(open('meds_on_grid_high_mimic.pickle','r'))
    baseline_covs = pickle.load(open('baseline_covs_high_mimic.pickle','r'))
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,
            labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

def prep_highf_mgp():
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    sub_stays_included = pickle.load(open('sub_stays_included.pickle','r'))
    sub_stay = defaultdict(list)
    for (sub,stay) in sub_stays_included:
        sub_stay[sub].append(stay)
    #subject_ids = subject_ids[:10]
    #cancelled_subs = []
    #subject_ids = [20, 107,194, 123, 160, 217, 292, 263, 125, 135, 33]
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
    #waveforms = []

    Ethnicity = {}
    eth_count = 0
    Gender = {'M':0, 'F':1}

    breakflag = False

    for sub in sub_stay.keys():
        print("Preparing subject %s"%str(sub))
        Y_i = []
        ind_kf_i = []
        ind_kt_i = []
        stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
        #timeline = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/episode1_timeseries.csv')
        intime = pd.to_datetime(stays['INTIME'])
        outtime = pd.to_datetime(stays['OUTTIME'])
        starttime = intime.dt.round('1h')
        label = stays['MORTALITY_INHOSPITAL'][0]
        for stay_no in sub_stay[sub]:
            try:
                timeline = pd.read_csv(data_path+'root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv')
            except:
                #print "no timeline"+str(sub)
                continue
            grid_times = list(np.arange(ceil(((outtime-starttime).dt.total_seconds()/(60*60))[stay_no])+1))
            if len(grid_times)>40:
                #cancelled_subs.append(sub)
                #print "no grid"+str(sub)
                continue
            timeline = timeline[timeline.Hours>=0]
            timeline = timeline.drop_duplicates()
            #timeline = timeline.fillna(0)
            if timeline.empty:
                continue
            try:
                wave = pd.read_csv(data_path+'waves/'+str(sub)+'_stay'+str(stay_no)+'.wav')
            except:
                #print "wave not"+str(sub)
                continue
            try:
                baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv', )
            except:
                #print "baseline"+str(sub)
                continue
            t_i = list(timeline.Hours)
            #add the time from waveforms as well as that will have to be sorted into the T and appended and removed for duplicates from this T_i
            wave_t = [(i+1)*0.1 for i in range((len(grid_times)-1)*6)]
            T_i = sorted(list(set(t_i).union(set(wave_t))))
            #for every 10 minutes
            gran = 125*60*10
            wavem = wave.rolling(gran).mean()
            wavem = wavem.iloc[::gran]
            #print wavem.head()
            wavestd = wave.rolling(gran).std()
            wavestd = wavestd.iloc[::gran]
            #this becomes param value for ind_kf??
            column_map = {n:i for i,n in enumerate(list(timeline.columns))}
            len_columns = len(timeline.columns)
            m_i = len_columns
            s_i = m_i+1
            t_index = 0
            timeline_index = 0
            for t in T_i:
                if t in t_i:
                    values = timeline.iloc[timeline_index]
                    presence = timeline.iloc[timeline_index].isnull()
                    for i in range(1,len_columns):
                        if presence[i]==False:
                            if type(values[i]) is not str:
                                if values[i]>0:
                                    Y_i.append(np.log(values[i]))
                                    ind_kf_i.append(i-1)
                                    ind_kt_i.append(t_index)
                    timeline_index += 1
                if t in wave_t:
                    index = int((t/0.1)-1)
                    vm = wavem.iloc[index]
                    vs = wavestd.iloc[index]
                    if not isnan(vm):
                        #print vm[0]
                        Y_i.append(vm[0])
                        ind_kf_i.append(m_i)
                        ind_kt_i.append(t_index)
                        Y_i.append(vs[0])
                        ind_kf_i.append(s_i)
                        ind_kt_i.append(t_index)
                t_index += 1
            if len(Y_i)>250:
                #print "too many"+str(sub)
                continue
            print "processing stay"+str(stay_no)
            sub_stays_included.append((sub,stay_no))
            rnn_grid_times.append(grid_times)
            #waveforms.append(waveform)
            end_times.append(len(rnn_grid_times[-1])-1)
            num_obs_times.append(timeline.count()[0])
            #num_obs_values.append(np.sum(timeline.count()[1:]))
            num_obs_values.append(len(Y_i))
            num_rnn_grid_times.append(len(rnn_grid_times[-1]))
            #rnn_grid_times.append(list(np.arange(num_rnn_grid_times[-1])))
            baseline = baseline.fillna(0)
            baseline_i = baseline.iloc[0].to_list()
            #print baseline_i
            #raw_input()
            try:
                medicines = pd.read_csv(data_path+'medicines/'+str(sub)+'_stay'+str(stay_no)+'.med')
                #medicines = medicines.fillna(0)
                meds_on_grid_i = medicines.to_numpy()
            except:
                meds_on_grid_i = np.zeros((int(num_rnn_grid_times[-1]),5))
            #validation
            if num_rnn_grid_times[-1]!=len(meds_on_grid_i):
                print "tafavat in sub", sub, str(num_rnn_grid_times[-1]), str(len(meds_on_grid_i))
            meds_on_grid.append(meds_on_grid_i.tolist())
            baseline_covs.append(baseline_i)
            Y.append(Y_i)
            ind_kf.append(ind_kf_i)
            ind_kt.append(ind_kt_i)
            T.append(T_i)
            labels.append(label)
            if len(labels)>=150:
                breakflag = True
                break
        if breakflag:
            print("dataset ends at %s"%sub)
            break
    pickle.dump(num_obs_times, open('num_obs_times_high_mimic.pickle','w'))
    pickle.dump(num_obs_values, open('num_obs_values_high_mimic.pickle','w'))
    pickle.dump(num_rnn_grid_times, open('num_rnn_grid_times_high_mimic.pickle','w'))
    pickle.dump(rnn_grid_times, open('rnn_grid_times_high_mimic.pickle','w'))
    pickle.dump(labels, open('labels_high_mimic.pickle','w'))
    pickle.dump(T, open('T_high_mimic.pickle','w'))
    pickle.dump(Y, open('Y_high_mimic.pickle','w'))
    pickle.dump(ind_kf, open('ind_kf_high_mimic.pickle','w'))
    pickle.dump(ind_kt, open('ind_kt_high_mimic.pickle','w'))
    pickle.dump(meds_on_grid, open('meds_on_grid_high_mimic.pickle','w'))
    pickle.dump(baseline_covs, open('baseline_covs_high_mimic.pickle','w'))
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

def retrieve_mimic_dataset():
    num_obs_times = pickle.load(open('num_obs_times_mimic.pickle','r'))
    num_obs_values = pickle.load(open('num_obs_values_mimic.pickle','r'))
    num_rnn_grid_times = pickle.load(open('num_rnn_grid_times_mimic.pickle','r'))
    rnn_grid_times = pickle.load(open('rnn_grid_times_mimic.pickle','r'))
    labels = pickle.load(open('labels_mimic.pickle','r'))
    T = pickle.load(open('T_mimic.pickle','r'))
    Y = pickle.load(open('Y_mimic.pickle','r'))
    ind_kf = pickle.load(open('ind_kf_mimic.pickle','r'))
    ind_kt = pickle.load(open('ind_kt_mimic.pickle','r'))
    meds_on_grid = pickle.load(open('meds_on_grid_mimic.pickle','r'))
    baseline_covs = pickle.load(open('baseline_covs_mimic.pickle','r'))
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,
            labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

def prep_baseline_mgp():
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    #subject_ids = subject_ids[:700]
    #cancelled_subs = []
    #subject_ids = [20, 107,194, 123, 160, 217, 292, 263, 125, 135, 33]
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

    breakflag = False

    for sub in subject_ids:
        print("Preparing subject %s"%str(sub))
        Y_i = []
        ind_kf_i = []
        ind_kt_i = []
        stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
        #timeline = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/episode1_timeseries.csv')
        intime = pd.to_datetime(stays['INTIME'])
        outtime = pd.to_datetime(stays['OUTTIME'])
        starttime = intime.dt.round('1h')
        label = stays['MORTALITY_INHOSPITAL'][0]
        for stay_no in range(stays.shape[0]):
            try:
                timeline = pd.read_csv(data_path+'root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv')
            except:
                continue
            grid_times = list(np.arange(ceil(((outtime-starttime).dt.total_seconds()/(60*60))[stay_no])+1))
            if len(grid_times)>45:
                #cancelled_subs.append(sub)
                continue
            timeline = timeline[timeline.Hours>=0]
            timeline = timeline.drop_duplicates()
            #timeline = timeline.fillna(0)
            if timeline.empty:
                continue
            T_i = timeline.Hours
            column_map = {n:i for i,n in enumerate(list(timeline.columns))}
            len_columns = len(timeline.columns)
            for t in range(T_i.shape[0]):
                values = timeline.iloc[t]
                #print values
                #values = np.log(values)
                #print values
                presence = timeline.iloc[t].isnull()
                #print presence
                for i in range(1,len_columns):
                    if presence[i]==False:
                        if type(values[i]) is not str:
                            if values[i]>0:
                                Y_i.append(np.log(values[i]))
                            else:
                                Y_i.append(values[i])
                            ind_kf_i.append(i-1)
                            ind_kt_i.append(t)
                #print Y_i
                #print ind_kf_i
                #print ind_kt_i
            #print timeline.head()
            if len(Y_i)>200:
                continue
            rnn_grid_times.append(grid_times)
            end_times.append(len(rnn_grid_times[-1])-1)
            num_obs_times.append(timeline.count()[0])
            #num_obs_values.append(np.sum(timeline.count()[1:]))
            num_obs_values.append(len(Y_i))
            num_rnn_grid_times.append(len(rnn_grid_times[-1]))
            #rnn_grid_times.append(list(np.arange(num_rnn_grid_times[-1])))
            try:
                baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv', )
            except:
                continue
            baseline = baseline.fillna(0)
            baseline_i = baseline.iloc[0].to_list()
            #print baseline_i
            #raw_input()
            try:
                medicines = pd.read_csv(data_path+'medicines/'+str(sub)+'_stay'+str(stay_no)+'.med')
                #medicines = medicines.fillna(0)
                meds_on_grid_i = medicines.to_numpy()
            except:
                meds_on_grid_i = np.zeros((int(num_rnn_grid_times[-1]),5))
            #validation
            if num_rnn_grid_times[-1]!=len(meds_on_grid_i):
                print "tafavat in sub", sub, str(num_rnn_grid_times[-1]), str(len(meds_on_grid_i))
            meds_on_grid.append(meds_on_grid_i.tolist())
            baseline_covs.append(baseline_i)
            Y.append(Y_i)
            ind_kf.append(ind_kf_i)
            ind_kt.append(ind_kt_i)
            T.append(T_i.tolist())
            labels.append(label)
            if len(labels)>=1000:
                breakflag = True
                break
        if breakflag:
            print("dataset ends at %s"%sub)
            break
    #print("num of grid times:%s"%num_rnn_grid_times)
    #'''
    print np.array(num_obs_times).mean()
    print np.array(num_obs_values).mean()
    print np.array(num_rnn_grid_times).mean()
    #print rnn_grid_times
    #print labels
    #print T
    #print "printing Y"
    #print Y
    #print "kf"
    #print ind_kf
    #print "kt"
    #print ind_kt
    #print "meds"
    #print meds_on_grid
    #print "baselines"
    #print baseline_covs
    #'''
    pickle.dump(num_obs_times, open('num_obs_times_mimic.pickle','w'))
    pickle.dump(num_obs_values, open('num_obs_values_mimic.pickle','w'))
    pickle.dump(num_rnn_grid_times, open('num_rnn_grid_times_mimic.pickle','w'))
    pickle.dump(rnn_grid_times, open('rnn_grid_times_mimic.pickle','w'))
    pickle.dump(labels, open('labels_mimic.pickle','w'))
    pickle.dump(T, open('T_mimic.pickle','w'))
    pickle.dump(Y, open('Y_mimic.pickle','w'))
    pickle.dump(ind_kf, open('ind_kf_mimic.pickle','w'))
    pickle.dump(ind_kt, open('ind_kt_mimic.pickle','w'))
    pickle.dump(meds_on_grid, open('meds_on_grid_mimic.pickle','w'))
    pickle.dump(baseline_covs, open('baseline_covs_mimic.pickle','w'))
    #'''
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

if __name__=="__main__":
    #prep_baseline_mgp()
    prep_highf_mgp()
