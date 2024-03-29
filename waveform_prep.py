import pandas as pd
import numpy as np
import os
from math import ceil
import wfdb
import pickle
import datetime
from collections import defaultdict

def isnat(dt):
    dtyp = str(dt.dtype)
    if 'datetime64' in dtyp or 'timedelta64' in dtyp:
        return dt.view('i8')==np.datetime64('NAT').view('i8')
    return False

def dataset_prep():
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    #print lines[:10]
    subject_ids = [int(x[0]) for x in lines]
    #subject_indices = [x[1] for x in lines]
    fwaves = open('/data/suparna/MatchedSubset_MIMIC3/RECORDS-waveforms','r')
    #create a dict of sub and the waveforms present
    waves = defaultdict(list)
    #sigs = []
    print("Reading records")

    fdone = open('waveforms_processed','r')
    subs_done = []
    for line in fdone:
        sub = int(line.split(' ')[2])
        subs_done.append(sub)
    subs_done = subs_done[:-1]
    fdone.close()
    for line in fwaves:
        line = line.strip('\n')
        path = os.path.join('/data/suparna/MatchedSubset_MIMIC3/',line)
        print("Processing %s"%path)
        s,date = line.split('/')[2].split('-',1)
        y,m,d,h,mi = map(int,date.split('-'))
        day = datetime.date(y,m,d)
        time = datetime.time(h,mi)
        sub = int(s[1:])
        if sub not in subject_ids:
            continue
        if sub in subs_done:
            continue
        waves[sub].append((day, time, path))
        '''
        try:
            __,fields = wfdb.rdsamp(path)
            for s in fields['sig_name']:
                if s not in sigs:
                    sigs.append(s)
        except ValueError:
            continue
        '''
    print("%s subjects to go"%(len(waves.keys())))
    #sigfile = open('various_signals.pickle','r')
    #pickle.dump(sigs, sigfile)
    #sigs= pickle.load(sigfile)
    #sigfile.close()
    #Only 'II' signal considered henceforth
    #sig_indices = {x:i for i,x in enumerate(sigs)}
    #sig_len = len(sigs)
    #sig_len = 1
    waves_subs = waves.keys()
    for sub in waves_subs:
        print("Processing subject %s"%(sub))
        stays = pd.read_csv('/data/suparna/MGP_data/root/'+str(sub)+'/stays.csv', parse_dates=True)
        stays['INTIME'] = pd.to_datetime(stays['INTIME'])
        stays['OUTTIME'] = pd.to_datetime(stays['OUTTIME'])
        for i in range(stays.shape[0]):
            try:
                paths_to_consider = []
                for date, time, path in waves[sub]:
                    base_time = datetime.datetime.combine(date,time)
                    if base_time>=stays['INTIME'][i] and base_time<=stays['OUTTIME'][i]:
                        paths_to_consider.append(path)
                #floor of start and ceil of end!! is the convention for all labs and vitals
                start_time = stays['INTIME'].dt.round('1h')
                #start = start_time[i]
                end_time = stays['OUTTIME'][i]
                total_hours = ceil(((stays['OUTTIME']-start_time).dt.total_seconds()/(60*60))[i])
                #print("intime:%s, outtime:%s"%(stays['OUTTIME'][i],stays['INTIME'][i]))
                total_rows = int(total_hours*3600*125)
                stay_timeline = np.empty((total_rows))
                stay_timeline[:] = np.nan
                for path in paths_to_consider:
                    signal, fields = wfdb.rdsamp(path)
                    try:
                        sig_index = fields['sig_name'].index('II')
                    except:
                        continue
                    signal = signal[:,sig_index]
                    print("signal head:%s"%signal[:5])
                    base_time = datetime.datetime.combine(fields['base_date'],fields['base_time'])
                    secs = signal.shape[0]/125
                    print("signal shape for %s:%s"%(path,signal.shape))
                    #end_time = base_time+datetime.timedelta(seconds=secs)
                    start_row = int(ceil((base_time-start_time)[i].total_seconds()*125))
                    end_row = start_row+signal.shape[0]
                    end_i = end_row
                    if start_row>=stay_timeline.shape[0]:
                        continue
                    if end_row>stay_timeline.shape[0]:
                        end_row = stay_timeline.shape[0]
                        end_i = end_row-start_row
                    print("[%s:%s][:%s]"%(start_row,end_row,end_i))
                    #signal_indices = [sig_indices[x] for x in fields['sig_name']]
                    #for i,j in enumerate(signal_indices):
                    #    stay_timeline[start_row:end_row,j] = signal[:end_i,i]
                    stay_timeline[start_row:end_row] = signal
                stay_pd = pd.DataFrame(data=stay_timeline, columns=['II'])
                stay_pd.to_csv(os.path.join('/data/suparna/MGP_data/waveforms/',str(sub)+"_stay"+str(i)+'.wav'), index=False)
                '''
                #signal = np.nan_to_num(signal)
                print(sub)
                if base_time>=stays['INTIME'][i] and base_time<=stays['OUTTIME'][i]:
                    postrows = int(floor((stays['OUTTIME']-base_time)[i].total_seconds()*125))
                    print("%s,%s,%s"%(base_time,stays['INTIME'][i],stays['OUTTIME'][i]))
                    #print prerows, postrows
                    pre = np.empty((prerows,signal.shape[1]))
                    pre[:] = np.nan
                    post = np.empty((postrows, signal.shape[1]))
                    post[:] = np.nan
                    signal = np.concatenate((pre,signal,post))
                else:
                    precut = 0
                    postcut = 0
                    if base_time<stays['INTIME'][i]:
                        precut = int(ceil((stays['INTIME']-base_time)[i].total_seconds()*125))
                    if base_time>stays['OUTTIME'][0]:
                        postcut = int(ceil((base_time-stays['OUTTIME'])[i].total_seconds()*125))
                    end_index = stays.shape[0]-postcut
                    print("%s,%s,%s"%(base_time,stays['INTIME'][i],stays['OUTTIME'][i]))
                    signal = signal[precut:end_index]
                    if signal.shape[1]<250:
                        continue
                '''
            except (ValueError, KeyError) as e:
                continue

if __name__=="__main__":
    dataset_prep()
