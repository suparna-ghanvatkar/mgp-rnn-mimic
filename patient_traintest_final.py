'''
Script to create a list of patients splits which follows the various criteria..
The waveforms need to be analyzed from their dump file
'''
import pandas as pd
import numpy as np
import numpy.ma as ma
from math import ceil, isnan
import pickle
import wfdb
import datetime
from time import time
from collections import defaultdict
import concurrent.futures
from multiprocessing import Pool
from pykalman import KalmanFilter

data_path = '/data/suparna/MGP_data/'
glascow_eye_open = {}
glascow_motor = {}
glascow_total = {}
glascow_verbal = {}
def get_encounter(values):
    (sub,stay_no),data_index= values
    sub = int(sub)
    stay_no = int(stay_no)
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
    #deleting these right now and checking
    """
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
    """
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
    starttime = starttime[stay_no]
    endtime = starttime+datetime.timedelta(hours=24)
    wavepath = '/data/suparna/MatchedSubset_MIMIC3/'
    waveform = np.empty((24*60*60*125))
    waveform[:] = np.nan
    #if len(data_index)>1:
    #    print("%s,%s"%(sub,stay_no))
    prev_end = 0
    prev_start = 0
    for date,index in data_index:
        subpathstr = 'p'+substr[:2]+'/p'+substr+'/p'+substr+'-'+date
        signal,fields = wfdb.rdsamp(wavepath+subpathstr, channels=[index])
        base_time = datetime.datetime.combine(fields['base_date'],fields['base_time'])
        start_row = int(ceil((base_time-starttime).total_seconds())*125)
        available_len = len(waveform)-start_row
        #if prev_end>start_row:
        #    if fabs(start_row-prev_start)>37500:    #i.e. if the gap between two recordings start time is greater than 5 minuets
        #        print("Problem in: %s,%s and has overlap of %s minutes"%(sub, stay_no,(prev_end-start_row)/(125*60)))
        #print(available_len)
        #if the waveform has less available length than the signal available then restrict else has no effect
        #print(fields['sig_len'])
        end_row = fields['sig_len']+start_row
        #if len(data_index)>1:
        #    print("%s,%s"%(start_row,end_row))
        #prev_end = end_row
        #prev_start = start_row
        if available_len<fields['sig_len']:
            waveform[start_row:] = signal[:available_len,0]
        else:
            waveform[start_row:end_row] = signal[:,0]
    #mask = np.isnan(waveform)
    #wave_mean = np.nanmean(waveform)
    #print(wave_mean)
    #waveform = np.nan_to_num(waveform, nan=wave_mean)
    waveform = np.where(np.isnan(waveform), ma.array(waveform, mask=np.isnan(waveform)).mean(),waveform)
    waveform = np.column_stack((np.mean(waveform.reshape(-1,125), axis=1), np.std(waveform.reshape(-1,125), axis=1)))
    #x = np.ma.masked_array(waveform, mask=mask)
    #initializing the kalman filter
    #kf = KalmanFilter(em_vars=['transition_covariance','observation_covariance'])
    #smoothed = kf.em(x, n_iter=2).smooth(x)[0]
    #masked_indices = np.where(mask)
    #newmask = np.invert(mask)
    #smoothed_values = np.ma.masked_array(smoothed, mask=newmask)
    #smoothed_values = smoothed_values[smoothed_values.mask==False]
    #np.put(waveform, masked_indices, smoothed_values)
    #print waveform.shape
    #print("done padding")
    baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv')
    baseline = baseline.fillna(0)
    baseline_i = baseline.iloc[0].to_list()
    try:
        medicines = pd.read_csv(data_path+'medicines/'+str(sub)+'_stay'+str(stay_no)+'.med',nrows=24)
        meds_on_grid_i = medicines.to_numpy()
    except:
        meds_on_grid_i = np.zeros((24,5))
    return T_i,Y_i,ind_kf_i,ind_kt_i,baseline_i,meds_on_grid_i,grid_times,label,waveform

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
    sub_stay = pickle.load(open('icis_revision/filtered_substays_'+train+'_fold'+str(fold)+'.pickle','rb'))
    #sub_stay = sub_stay[:60]
    tot = len(sub_stay)
    #sub_stay = sub_stay[:(tot/5)*5]
    print("Preparing dataset of size:%s"%(tot))
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


    breakflag = False
    stime = time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #for (sub,stay_no),date_index in sub_stay:
        results = executor.map(get_encounter, sub_stay)
        for stayinfo,values in zip(sub_stay,results):
            (sub,stay_no),date_index = stayinfo
            #values = get_encounter(((sub,stay_no),date_index))
            start_time = time()
            #print("Preparing subject %s"%str(sub))
            T_i,Y_i,ind_kf_i,ind_kt_i,baseline_i,meds_on_grid_i,grid_times,label,waveform= values
            rnn_grid_times.append(grid_times)
            waveforms.append(waveform)
            end_times.append(len(rnn_grid_times[-1])-1)
            num_obs_times.append(len(T_i))
            num_obs_values.append(len(Y_i))
            num_rnn_grid_times.append(len(rnn_grid_times[-1]))
            #validation
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
    print(np.array(num_obs_times).mean())
    print(np.array(num_obs_values).mean())
    print(np.array(num_rnn_grid_times).mean())
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
    prep_mimic('train',0)
