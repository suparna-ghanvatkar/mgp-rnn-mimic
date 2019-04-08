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

data_path = '/data/suparna/MGP_data/'

def prep_mimic(train):
    '''
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    sub_stays_included = []
    '''
    sub_stay = pickle.load(open('final_substays.pickle','r'))
    sub_stay = sub_stay[:500]
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

    Ethnicity = {}
    eth_count = 0
    Gender = {'M':0, 'F':1}
    #for generating the discrete numeric lables for glascow columns
    glascow_eye_open = {}
    glascow_motor = {}
    glascow_total = {}
    glascow_verbal = {}

    breakflag = False
    stime = time()
    for sub,stay_no,date,index in sub_stay:
        start_time = time()
        print("Preparing subject %s"%str(sub))
        Y_i = []
        ind_kf_i = []
        ind_kt_i = []
        stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
        #timeline = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/episode1_timeseries.csv')
        intime = pd.to_datetime(stays['INTIME'])
        outtime = pd.to_datetime(stays['OUTTIME'])
        starttime = intime.dt.round('1h')
        label = stays['MORTALITY_INHOSPITAL'][stay_no]
        #for stay_no in range(stays.shape[0]):
        try:
            timeline = pd.read_csv(data_path+'root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv')
        except:
            print "no timeline"+str(sub)
            continue
        #grid_times = list(np.arange(ceil(((outtime-starttime).dt.total_seconds()/(60*60))[stay_no])+1))
        grid_times = range(24)
        if len(grid_times)<24:
            #cancelled_subs.append(sub)
            #print "no grid"+str(sub)
            continue
        timeline = timeline[timeline.Hours>=0]
        timeline = timeline[timeline.Hours<=24]
        timeline = timeline.drop_duplicates()
        #timeline = timeline.fillna(0)
        if timeline.empty:
            continue
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
        #timeline = timeline.apply(np.log)
        #print timeline.shape
        #imp = Imputer(strategy="mean",axis=0)
        #timeline = scale(imp.fit_transform(timeline))
        #print timeline.shape, len_columns, T_i.shape
        #timeline = scaler.transform(imp.fit_transform(timeline))
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
        '''
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
        '''
        #if len(Y_i)>200:
            #print "too many"+str(sub)
            #continue
        #try:
            #wave = pd.read_csv(data_path+'waves/'+str(sub)+'_stay'+str(stay_no)+'.wav',nrows=len(grid_times)*60*60*125)
        substr = "%06d"%(sub)
        subpathstr = 'p'+substr[:2]+'/p'+substr+'/p'+substr+'-'+date
        wavepath = '/data/suparna/MatchedSubset_MIMIC3/'
        #print wavepath+subpathstr
        signal,fields = wfdb.rdsamp(wavepath+subpathstr, channels=[index])
        #except:
        #    print("wave not for %s,%s,%s,%s"%(sub,stay_no,date,index))
        #    continue
        try:
            baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv', )
        except:
            print "baseline"+str(sub)
            continue
        #print "processing stay"+str(stay_no)
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
        #print waveform[-135:-125]
        #waveform = wave.rolling(125).agg({'M':'mean','S':'std'})
        #waveform = waveform[::125]
        #waveform = waveform.fillna(0)
        #print waveform.head()
        #waveform = waveform.to_numpy()
        #print("converted to numpy")
        #print waveform.shape
        waveform = np.pad(waveform,((start_row,(len(grid_times)*60*60)-end_row),(0,0)),'constant')
        print("done padding")
        #print waveform.shape
        #if waveform.shape[1]!=2:

        '''
        waveform = np.zeros((len(grid_times)*60*60,2))
        nrows = waveform.shape[0]
        wavem = wave.rolling(125).mean()
        wavem = wavem.fillna(0)
        wavem = wavem.iloc[::125]
        #wavem = wavem[:nrows]
        #print wavem.head()
        wavem = wavem.to_numpy()
        #if wavem.shape[0]<waveform.shape[0]:
        #    nrows = wavem.shape[0]
        #print wavem.shape
        waveform[:nrows,0] =wavem[:nrows,0]
        #wavem = np.pad(wavem, (0,(len(grid_times)*60*60)-len(wavem)),'constant')
        #print wavem.shape
        wavestd = wave.rolling(125).std()
        wavestd = wavestd.fillna(0)
        wavestd = wavestd.iloc[::125]
        wavestd = wavestd.to_numpy()
        waveform[:nrows,1] =wavestd[:nrows,0]
        #print waveform[:5]
        #print wavestd.shape
        #wavestd = np.pad(wavestd, (0,(len(grid_times)*60*60)-len(wavestd)),'constant')
        #print wavestd.shape
        #waveform = np.column_stack((wavem,wavestd))
        #print waveform.shape
        #waveform = np.pad(waveform, (0,len(grid_times)*60*60), 'constant')
        '''
        if waveform.shape[0]!=(len(grid_times)*60*60):
            print "tafavat in waveform for sub"+str(sub)+str(waveform.shape)+" : "+str(len(grid_times)*60*60)
        #sub_stays_included.append((sub,stay_no))
        #print("starting to append")
        rnn_grid_times.append(grid_times)
        waveforms.append(waveform)
        end_times.append(len(rnn_grid_times[-1])-1)
        num_obs_times.append(len(T_i))
        #num_obs_values.append(np.sum(timeline.count()[1:]))i
        #print Y_i
        num_obs_values.append(len(Y_i))
        num_rnn_grid_times.append(len(rnn_grid_times[-1]))
        #rnn_grid_times.append(list(np.arange(num_rnn_grid_times[-1])))
        baseline = baseline.fillna(0)
        baseline_i = baseline.iloc[0].to_list()
        #print baseline_i
        #raw_input()
        try:
            medicines = pd.read_csv(data_path+'medicines/'+str(sub)+'_stay'+str(stay_no)+'.med',nrows=24)
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
        end_time = time()
        print("took time %s"%(end_time-start_time))
        #if len(labels)>=200:
        #    breakflag = True
        #    break
        if breakflag:
            print("dataset ends at %s"%sub)
            break
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
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs,waveforms)

if __name__=="__main__":
    prep_mimic('train')
