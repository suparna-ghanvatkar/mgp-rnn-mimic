import pandas as pd
import numpy as np
from math import ceil, isnan
import pickle
from collections import defaultdict
import wfdb
from time import time
import datetime
from sklearn.preprocessing import StandardScaler, Imputer, scale
import concurrent.futures

'''
The baselines are: Ethnicity, Gender, Age, Height, Weight
'''
glascow_eye_open = {}
glascow_motor = {}
glascow_total = {}
glascow_verbal = {}

data_path = '/data/suparna/MGP_data/'

def retrieve_high_mimic_dataset(train):
    num_obs_times = pickle.load(open('num_obs_times_high_'+train+'_mimic.pickle','r'))
    num_obs_values = pickle.load(open('num_obs_values_high_'+train+'_mimic.pickle','r'))
    num_rnn_grid_times = pickle.load(open('num_rnn_grid_times_high_'+train+'_mimic.pickle','r'))
    rnn_grid_times = pickle.load(open('rnn_grid_times_high_'+train+'_mimic.pickle','r'))
    labels = pickle.load(open('labels_high_'+train+'_mimic.pickle','r'))
    T = pickle.load(open('T_high_'+train+'_mimic.pickle','r'))
    Y = pickle.load(open('Y_high_'+train+'_mimic.pickle','r'))
    ind_kf = pickle.load(open('ind_kf_high_'+train+'_mimic.pickle','r'))
    ind_kt = pickle.load(open('ind_kt_high_'+train+'_mimic.pickle','r'))
    meds_on_grid = pickle.load(open('meds_on_grid_high_'+train+'_mimic.pickle','r'))
    baseline_covs = pickle.load(open('baseline_covs_high_'+train+'_mimic.pickle','r'))
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,
            labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

def get_encounter_high(values):
    sub,stay_no,date,index = values
    Y_i = []
    ind_kf_i = []
    ind_kt_i = []
    stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
    #timeline = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/episode1_timeseries.csv')
    intime = pd.to_datetime(stays['INTIME'])
    outtime = pd.to_datetime(stays['OUTTIME'])
    starttime = intime.dt.round('1h')
    label = stays['MORTALITY_INHOSPITAL'][stay_no]
    timeline = pd.read_csv(data_path+'root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv')
    grid_times = range(24)
    #grid_times = list(np.arange(ceil(((outtime-starttime).dt.total_seconds()/(60*60))[stay_no])+1))
    #if len(grid_times)<24:
        #cancelled_subs.append(sub)
        #print "no grid"+str(sub)
        #continue
    timeline = timeline[timeline.Hours>=0]
    timeline = timeline[timeline.Hours<=24]
    timeline = timeline.drop_duplicates()
    #timeline = timeline.fillna(0)
    #try:
    #    wave = pd.read_csv(data_path+'waves/'+str(sub)+'_stay'+str(stay_no)+'.wav')
    #except:
    #    print "wave not"+str(sub)
    #    continue
    substr = "%06d"%(sub)
    subpathstr = 'p'+substr[:2]+'/p'+substr+'/p'+substr+'-'+date
    wavepath = '/data/suparna/MatchedSubset_MIMIC3/'
    #print wavepath+subpathstr
    signal,fields = wfdb.rdsamp(wavepath+subpathstr, channels=[index])
    baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv', )
    t_i = timeline.Hours
    #print signal.shape
    #add the time from waveforms as well as that will have to be sorted into the T and appended and removed for duplicates from this T_i
    wave_t = [i*1.667 for i in range(len(grid_times)*6)]
    T_i = sorted(list(set(t_i).union(set(wave_t))))
    #creating map for the timeline hours into the final time indices
    T_i_map = {n:i for i,n in enumerate(T_i)}
    t_i_map = {i:T_i_map[n] for i,n in enumerate(t_i)}
    wave_t_map = {i:T_i_map[n] for i,n in enumerate(wave_t)}
    row_map = {n:i for i,n in enumerate(t_i.index)}
    #for every 10 minutes
    gran = 125*60*10
    starttime = intime[stay_no]
    endtime = starttime+datetime.timedelta(hours=24)
    base_time = datetime.datetime.combine(fields['base_date'],fields['base_time'])
    if base_time>starttime:
        start_row = int(ceil((base_time-starttime).total_seconds())//600)
    else:
        start_row = 0
    #print start_row
    signal = np.pad(signal,((0,(int(ceil(len(signal)/(125.0*600))*gran))-len(signal)),(0,0)), 'constant',constant_values=(np.nan))
    #print signal.shape
    end_row = start_row+(signal.shape[0]/gran)
    last_row = 24*6
    if last_row<end_row:
        end_row = last_row
        signal = signal[:(last_row-start_row)*gran]
    waveform = np.column_stack((np.mean(signal.reshape(-1,gran), axis=1), np.std(signal.reshape(-1,gran), axis=1)))
    #this becomes param value for ind_kf??
    column_map = {n:i for i,n in enumerate(list(timeline.columns))}
    #create and add discrete numeric values for glascow records
    col_names = [('Glascow coma scale eye opening',glascow_eye_open), ('Glascow coma scale motor response',glascow_motor), ('Glascow coma scale total',glascow_total),('Glascow coma scale verbal response',glascow_verbal)]
    for col,dlist in col_names:
        tseries = timeline[col]
        tseries = tseries.dropna()
        for index, value in tseries.iteritems():
            try:
                value = dlist[value]
            except:
                dlist[value] = len(dlist)
                value = dlist[value]
            Y_i.append(value)
            ind_kf_i.append(column_map[col]-1)
            ind_kt_i.append(t_i_map[row_map[index]])
    #drop the glascow and hours column and create a mask of values present
    col_del = ['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response']
    timeline = timeline.drop(col_del,axis=1)
    mask = timeline.notnull()
    dropped_col_map = {i:column_map[n] for i,n in enumerate(list(timeline.columns))}
    #add values to Y,ind_kf and ind_kt acc to the mask
    len_columns = len(timeline.columns)
    m_i = len_columns
    s_i = m_i+1
    for t in range(len(t_i)):
        presence = mask.iloc[t]
        #print presence
        for i in range(len_columns):
            value = timeline.iloc[t][i]
            if presence[i]==True and type(value) is not str:
                Y_i.append(value)
                ind_kf_i.append(dropped_col_map[i]-1)
                ind_kt_i.append(t_i_map[t])
    for index in range(start_row,end_row):
        vm = waveform[index-start_row][0]
        vs = waveform[index-start_row][1]
        if not isnan(vm):
            Y_i.append(vm)
            ind_kf_i.append(m_i)
            ind_kt_i.append(wave_t_map[index])
            Y_i.append(vs)
            ind_kf_i.append(s_i)
            ind_kt_i.append(wave_t_map[index])
    baseline = baseline.fillna(0)
    baseline_i = baseline.iloc[0].to_list()
    #print baseline_i
    #raw_input()
    try:
        medicines = pd.read_csv(data_path+'medicines/'+str(sub)+'_stay'+str(stay_no)+'.med',nrows=24)
        #medicines = medicines.fillna(0)
        meds_on_grid_i = medicines.to_numpy()
    except:
        meds_on_grid_i = np.zeros((24,5))
    return (T_i,Y_i,ind_kf_i,ind_kt_i,baseline_i,meds_on_grid_i,grid_times,label)

def prep_highf_mgp(train,fold):
    '''
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
    '''
    #sub_stay = pickle.load(open('sub_stay_'+train+'_mimic.pickle','r'))
    #sub_stay = sub_stay[:10]
    sub_stay = pickle.load(open('final_substays_'+train+'_'+str(fold)+'.pickle','r'))
    count = len(sub_stay)
    sub_stay = sub_stay[:(count/5)*5]
    #sub_stay = sub_stay[:5000]
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
    #for generating the discrete numeric lables for glascow columns
    glascow_eye_open = {}
    glascow_motor = {}
    glascow_total = {}
    glascow_verbal = {}
    breakflag = False

    with concurrent.futures.ProcessPoolExecutor() as executor:
        #for sub,stay_no,date,index in sub_stay:
        results = executor.map(get_encounter_high, sub_stay,chunksize=3)
        for stay_info,values in zip(sub_stay,results):
            sub,stay_no,date,index = stay_info
            #print("Preparing subject %s"%str(sub))
            T_i,Y_i,ind_kf_i,ind_kt_i,baseline_i,meds_on_grid_i,grid_times,label = values
            rnn_grid_times.append(grid_times)
            end_times.append(len(rnn_grid_times[-1])-1)
            num_obs_times.append(len(T_i))
            num_obs_values.append(len(Y_i))
            num_rnn_grid_times.append(len(rnn_grid_times[-1]))
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
            end_time = time()
            #print("took time:%s"%(end_time-start_time))
    '''
    print np.array(num_obs_times).mean()
    print np.array(num_obs_values).mean()
    print np.array(num_rnn_grid_times).mean()
    print rnn_grid_times
    print labels
    print T
    print len(T)
    print "printing Y"
    print Y
    print "kf"
    print ind_kf
    print "kt"
    print ind_kt
    print "meds"
    print meds_on_grid
    print "baselines"
    print baseline_covs
    #'''
    '''
    pickle.dump(num_obs_times, open('num_obs_times_high_'+train+'_mimic.pickle','w'))
    pickle.dump(num_obs_values, open('num_obs_values_high_'+train+'_mimic.pickle','w'))
    pickle.dump(num_rnn_grid_times, open('num_rnn_grid_times_high_'+train+'_mimic.pickle','w'))
    pickle.dump(rnn_grid_times, open('rnn_grid_times_high_'+train+'_mimic.pickle','w'))
    pickle.dump(labels, open('labels_high_'+train+'_mimic.pickle','w'))
    pickle.dump(T, open('T_high_'+train+'_mimic.pickle','w'))
    pickle.dump(Y, open('Y_high_'+train+'_mimic.pickle','w'))
    pickle.dump(ind_kf, open('ind_kf_high_'+train+'_mimic.pickle','w'))
    pickle.dump(ind_kt, open('ind_kt_high_'+train+'_mimic.pickle','w'))
    pickle.dump(meds_on_grid, open('meds_on_grid_high_'+train+'_mimic.pickle','w'))
    pickle.dump(baseline_covs, open('baseline_covs_high_'+train+'_mimic.pickle','w'))
    '''
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

def retrieve_mimic_dataset(train):
    num_obs_times = pickle.load(open('num_obs_times_'+train+'_mimic.pickle','r'))
    num_obs_values = pickle.load(open('num_obs_values_'+train+'_mimic.pickle','r'))
    num_rnn_grid_times = pickle.load(open('num_rnn_grid_times_'+train+'_mimic.pickle','r'))
    rnn_grid_times = pickle.load(open('rnn_grid_times_'+train+'_mimic.pickle','r'))
    labels = pickle.load(open('labels_'+train+'_mimic.pickle','r'))
    T = pickle.load(open('T_'+train+'_mimic.pickle','r'))
    Y = pickle.load(open('Y_'+train+'_mimic.pickle','r'))
    ind_kf = pickle.load(open('ind_kf_'+train+'_mimic.pickle','r'))
    ind_kt = pickle.load(open('ind_kt_'+train+'_mimic.pickle','r'))
    meds_on_grid = pickle.load(open('meds_on_grid_'+train+'_mimic.pickle','r'))
    baseline_covs = pickle.load(open('baseline_covs_'+train+'_mimic.pickle','r'))
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,
            labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

def get_encounter_baseline(values):
    sub,stay_no,date,index = values
    Y_i = []
    ind_kf_i = []
    ind_kt_i = []
    stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
    intime = pd.to_datetime(stays['INTIME'])
    outtime = pd.to_datetime(stays['OUTTIME'])
    starttime = intime.dt.round('1h')
    label = stays['MORTALITY_INHOSPITAL'][stay_no]
    timeline = pd.read_csv(data_path+'root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv')
    timeline = timeline[timeline.Hours>=0]
    timeline = timeline[timeline.Hours<=24]
    timeline = timeline.drop_duplicates()
    baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv', )
    grid_times = range(24)
    T_i = timeline.Hours
    column_map = {n:i for i,n in enumerate(list(timeline.columns))}
    #the drop and remove neg Hours value screws up the indices. So create a map for the final index numbers in the data frame and the actually 't' index we want to index. useful to glascow vars
    row_map = {n:i for i,n in enumerate(T_i.index)}
    #create and add discrete numeric values for glascow records
    col_names = [('Glascow coma scale eye opening',glascow_eye_open), ('Glascow coma scale motor response',glascow_motor), ('Glascow coma scale total',glascow_total),('Glascow coma scale verbal response',glascow_verbal)]
    for col,dlist in col_names:
        tseries = timeline[col]
        tseries = tseries.dropna()
        for index, value in tseries.iteritems():
            try:
                value = dlist[value]
            except:
                dlist[value] = len(dlist)
                value = dlist[value]
            Y_i.append(value)
            ind_kf_i.append(column_map[col]-1)
            ind_kt_i.append(row_map[index])
    #drop the glascow and hours column and create a mask of values present
    col_del = ['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response']
    timeline = timeline.drop(col_del,axis=1)
    mask = timeline.notnull()
    dropped_col_map = {i:column_map[n] for i,n in enumerate(list(timeline.columns))}
    #log transform, impute and standardscaler
    len_columns = len(timeline.columns)
    #add values to Y,ind_kf and ind_kt acc to the mask
    for t in range(T_i.shape[0]):
        presence = mask.iloc[t]
        for i in range(len_columns):
            value = timeline.iloc[t][i]
            if presence[i]==True and type(value) is not str:
                Y_i.append(value)
                ind_kf_i.append(dropped_col_map[i]-1)
                ind_kt_i.append(t)
    baseline = baseline.fillna(0)
    baseline_i = baseline.iloc[0].to_list()
    try:
        medicines = pd.read_csv(data_path+'medicines/'+str(sub)+'_stay'+str(stay_no)+'.med',nrows=24)
        meds_on_grid_i = medicines.to_numpy()
    except:
        meds_on_grid_i = np.zeros((24,5))
    return T_i,Y_i,ind_kf_i,ind_kt_i,baseline_i,meds_on_grid_i,grid_times,label

def prep_baseline_mgp(train,fold):
    '''
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    sub_stay = defaultdict(list)
    '''
    #sub_stay = pickle.load(open('sub_stay_'+train+'_mimic.pickle','r'))
    #sub_stay = sub_stay[:10]
    sub_stay = pickle.load(open('final_substays_'+train+'_'+str(fold)+'.pickle','r'))
    count = len(sub_stay)
    sub_stay = sub_stay[:(count/5)*5]
    #sub_stay = sub_stay[:30]
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
    #for generating the discrete numeric lables for glascow columns

    breakflag = False

    with concurrent.futures.ProcessPoolExecutor() as executor:
        #for sub,stay_no,date,index in sub_stay:
        results = executor.map(get_encounter_baseline, sub_stay,chunksize=3)
        for stay_info,values in zip(sub_stay,results):
            sub,stay_no,date,index = stay_info
            T_i,Y_i,ind_kf_i,ind_kt_i,baseline_i,meds_on_grid_i,grid_times,label = values
            rnn_grid_times.append(grid_times)
            end_times.append(len(rnn_grid_times[-1])-1)
            num_obs_times.append(len(T_i))
            num_obs_values.append(len(Y_i))
            num_rnn_grid_times.append(len(rnn_grid_times[-1]))
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
            end_time = time()
    #print("num of grid times:%s"%num_rnn_grid_times)
    #'''
    print np.array(num_obs_times).mean()
    print np.array(num_obs_values).mean()
    print np.array(num_rnn_grid_times).mean()
    '''
    print rnn_grid_times
    print labels
    print T
    print "printing Y"
    print Y
    print "kf"
    print ind_kf
    print "kt"
    print ind_kt
    print "meds"
    print meds_on_grid
    print "baselines"
    print baseline_covs
    #'''
    #pickle.dump(sub_stay,open('sub_stay_mimic.pickle','w'))
    '''
    pickle.dump(num_obs_times, open('num_obs_times_'+train+'_mimic.pickle','w'))
    pickle.dump(num_obs_values, open('num_obs_values_'+train+'_mimic.pickle','w'))
    pickle.dump(num_rnn_grid_times, open('num_rnn_grid_times_'+train+'_mimic.pickle','w'))
    pickle.dump(rnn_grid_times, open('rnn_grid_times_'+train+'_mimic.pickle','w'))
    pickle.dump(labels, open('labels_'+train+'_mimic.pickle','w'))
    pickle.dump(T, open('T_'+train+'_mimic.pickle','w'))
    pickle.dump(Y, open('Y_'+train+'_mimic.pickle','w'))
    pickle.dump(ind_kf, open('ind_kf_'+train+'_mimic.pickle','w'))
    pickle.dump(ind_kt, open('ind_kt_'+train+'_mimic.pickle','w'))
    pickle.dump(meds_on_grid, open('meds_on_grid_'+train+'_mimic.pickle','w'))
    pickle.dump(baseline_covs, open('baseline_covs_'+train+'_mimic.pickle','w'))
    #'''
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)

if __name__=="__main__":
    #prep_baseline_mgp('train',0)
    prep_highf_mgp('train',0)
