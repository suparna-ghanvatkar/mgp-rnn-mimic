'''
Script to create a list of patients splits which follows the various criteria..
The waveforms need to be analyzed from their dump file
'''
import pandas as pd
import numpy as np
from math import ceil, isnan
import pickle
import wfdb
import datetime
from time import time
from collections import defaultdict
import concurrent.futures
from multiprocessing import Pool

data_path = '/data/suparna/MGP_data/'
glascow_eye_open = {}
glascow_motor = {}
glascow_total = {}
glascow_verbal = {}
def get_encounter(values):
    sub,stay_no,date,index,index2 = values
    Y_i = []
    ind_kf_i = []
    ind_kt_i = []
    stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
    intime = pd.to_datetime(stays['INTIME'])
    outtime = pd.to_datetime(stays['OUTTIME'])
    starttime = intime.dt.round('1h')
    label = stays['MORTALITY_INHOSPITAL'][stay_no]
    timeline = pd.read_csv(data_path+'root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv')
    grid_times = range(24)
    #timeline = timeline[['Hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response','Glucose','Heart Rate','Height','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure','Temperature','Weight','pH']]
    timeline = timeline[timeline.Hours>=0]
    timeline = timeline[timeline.Hours<=24]
    timeline = timeline.drop_duplicates()
    T_i = timeline.Hours
    column_map = {n:i for i,n in enumerate(list(timeline.columns))}
    #the drop and remove neg Hours value screws up the indices. So create a map for the final index numbers in the data frame and the actually 't' index we want to index. useful to glascow vars
    row_map = {n:i for i,n in enumerate(T_i.index)}
    #create and add discrete numeric values for glascow records
    col_names = [('Glascow coma scale eye opening',glascow_eye_open), ('Glascow coma scale motor response',glascow_motor), ('Glascow coma scale total',glascow_total),('Glascow coma scale verbal response',glascow_verbal)]
    for col,dlist in col_names:
        tseries = timeline[col]
        tseries = tseries.dropna()
        for ind, value in tseries.iteritems():
            try:
                value = dlist[value]
            except:
                dlist[value] = len(dlist)
                value = dlist[value]
            Y_i.append(value)
            ind_kf_i.append(column_map[col]-1)
            ind_kt_i.append(row_map[ind])
    #drop the glascow and hours column and create a mask of values present
    #print timeline.shape
    col_del = ['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response']
    timeline = timeline.drop(col_del,axis=1)
    #print timeline.shape
    mask = timeline.notnull()
    dropped_col_map = {i:column_map[n] for i,n in enumerate(list(timeline.columns))}
    #log transform, impute and standardscaler
    len_columns = len(timeline.columns)
    #add values to Y,ind_kf and ind_kt acc to the mask
    for t in range(T_i.shape[0]):
        presence = mask.iloc[t]
        #print presence
        for i in range(len_columns):
            value = timeline.iloc[t][i]
            if presence[i]==True and type(value) is not str:
                Y_i.append(value)
                ind_kf_i.append(dropped_col_map[i]-1)
                ind_kt_i.append(t)
    substr = "%06d"%(sub)
    subpathstr = 'p'+substr[:2]+'/p'+substr+'/p'+substr+'-'+date
    wavepath = '/data/suparna/MatchedSubset_MIMIC3/'
    #print wavepath+subpathstr
    signal,fields = wfdb.rdsamp(wavepath+subpathstr, channels=[index])
    baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv', )
    starttime = starttime[stay_no]
    endtime = starttime+datetime.timedelta(hours=24)
    base_time = datetime.datetime.combine(fields['base_date'],fields['base_time'])
    start_row = int(ceil((base_time-starttime).total_seconds()))
    signal = np.pad(signal,((0,(int(ceil(len(signal)/125.0)*125))-len(signal)),(0,0)), 'constant',constant_values=(np.nan))
    end_row = start_row+(signal.shape[0]/125)
    last_row = 24*60*60
    if last_row<end_row:
        end_row = last_row
        signal = signal[:(last_row-start_row)*125]
    waveform = np.column_stack((np.mean(signal.reshape(-1,125), axis=1), np.std(signal.reshape(-1,125), axis=1)))
    waveform = np.nan_to_num(waveform)
    waveform = np.pad(waveform,((start_row,(len(grid_times)*60*60)-end_row),(0,0)),'constant')
    #print("done padding")
    if index2!=-1:
        #wavepath = '/data/suparna/MatchedSubset_MIMIC3/'
        #print wavepath+subpathstr
        signal,fields = wfdb.rdsamp(wavepath+subpathstr, channels=[index2])
        #starttime = starttime[stay_no]
        #endtime = starttime+datetime.timedelta(hours=24)
        base_time = datetime.datetime.combine(fields['base_date'],fields['base_time'])
        start_row = int(ceil((base_time-starttime).total_seconds()))
        signal = np.pad(signal,((0,(int(ceil(len(signal)/125.0)*125))-len(signal)),(0,0)), 'constant',constant_values=(np.nan))
        end_row = start_row+(signal.shape[0]/125)
        last_row = 24*60*60
        if last_row<end_row:
            end_row = last_row
            signal = signal[:(last_row-start_row)*125]
        waveform2 = np.column_stack((np.mean(signal.reshape(-1,125), axis=1), np.std(signal.reshape(-1,125), axis=1)))
        waveform2 = np.nan_to_num(waveform2)
        waveform2 = np.pad(waveform2,((start_row,(len(grid_times)*60*60)-end_row),(0,0)),'constant')
    else:
        waveform2 = np.empty((int((endtime-starttime).total_seconds()),2))
    baseline = baseline.fillna(0)
    baseline_i = baseline.iloc[0].to_list()
    try:
        medicines = pd.read_csv(data_path+'medicines/'+str(sub)+'_stay'+str(stay_no)+'.med',nrows=24)
        meds_on_grid_i = medicines.to_numpy()
    except:
        meds_on_grid_i = np.zeros((24,5))
    return T_i,Y_i,ind_kf_i,ind_kt_i,baseline_i,meds_on_grid_i,grid_times,label,waveform,waveform2

def prep_mimic(train,fold):
    '''
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    sub_stays_included = []
    '''
    #sub_stay = pickle.load(open('final_substays_'+train+'_'+str(fold)+'.pickle','r'))
    sub_stay_ecg = pickle.load(open('balanced_data_'+train+'_'+str(fold)+'.pickle','r'))
    abp_ind = pickle.load(open('abp_index.pickle','r'))
    sub_stay = []
    count = 0
    for (sub,stay,date,ind) in sub_stay_ecg:
        try:
            ind2 = abp_ind[(sub,date)]
            count +=1
        except:
            ind2 = -1
        sub_stay.append((sub,stay,date,ind,ind2))
    print("ABP present in %s patients"%count)
    #sub_stay = sub_stay[:30]
    print sub_stay
    tot = len(sub_stay)
    #sub_stay = sub_stay[:(tot/5)*5]
    #print sub_stay
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
    waveforms = []
    waveforms2 = []

    Ethnicity = {}
    eth_count = 0
    Gender = {'M':0, 'F':1}
    #for generating the discrete numeric lables for glascow columns


    breakflag = False
    stime = time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #for sub,stay_no,date,index,ind2 in sub_stay:
        results = executor.map(get_encounter, sub_stay, chunksize=3)
        for stayinfo,values in zip(sub_stay,results):
            sub,stay_no,date,index,ind2 = stayinfo
            #values = get_encounter((sub,stay_no,date,index,ind2))
            start_time = time()
            #print("Preparing subject %s"%str(sub))
            T_i,Y_i,ind_kf_i,ind_kt_i,baseline_i,meds_on_grid_i,grid_times,label,waveform,waveform2 = values
            rnn_grid_times.append(grid_times)
            waveforms.append(waveform)
            waveforms2.append(waveform2)
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
            #print("took time %s"%(end_time-start_time))
    #print("num of grid times:%s"%num_rnn_grid_times)
    #'''
    etime = time()
    print("Total time for data prep: %s"%(etime-stime))
    print np.array(num_obs_times).mean()
    print np.array(num_obs_values).mean()
    print np.array(num_rnn_grid_times).mean()
    #print Y
    #print "storing the data"
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
    '''
    #pickle.dump(sub_stays_included, open('sub_stays_included.pickle','w'))
    '''
    pickle.dump(num_obs_times, open('num_obs_times_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(num_obs_values, open('num_obs_values_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(num_rnn_grid_times, open('num_rnn_grid_times_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(rnn_grid_times, open('rnn_grid_times_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(labels, open('labels_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(T, open('T_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(Y, open('Y_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(ind_kf, open('ind_kf_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(ind_kt, open('ind_kt_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(meds_on_grid, open('meds_on_grid_hierarchical_'+train+'_mimic.pickle','w'))
    pickle.dump(baseline_covs, open('baseline_covs_hierarchical_'+train+'_mimic.pickle','w'))
    #pickle.dump(waveforms, open('waveforms_hierarchical_'+train+'_mimic.pickle','w'))
    #'''
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs,waveforms,waveforms2)

if __name__=="__main__":
    prep_mimic('train',0)
