import pandas as pd
import numpy as np
from math import ceil

def getcvseries(med, intime, outtime, medmap):
    rnn_grid_times = np.arange(ceil(((outtime-intime).dt.total_seconds()/(60*60))[0])+1)
    medtimes = pd.to_datetime(med.CHARTTIME)
    medt = medtimes.apply(lambda x: x-intime)[0]
    medt = medt.dt.total_seconds()/3600
    med.CHARTTIME = medt
    med = med[med.CHARTTIME>=0]
    med = med.sort_values('CHARTTIME')
    medtimes = med.CHARTTIME.unique()
    meds = np.empty((len(rnn_grid_times),len(medmap.keys())))

    return meds

def getmvseries(med, intime, outtime, medmap):
    rnn_grid_times = np.arange(ceil(((outtime-intime).dt.total_seconds()/(60*60))[0])+1)
    medtimes = pd.to_datetime(med.STARTTIME)
    medt = medtimes.apply(lambda x: x-intime)[0]
    medt = medt.dt.total_seconds()/3600
    med.STARTTIME = medt
    med = med[med.STARTTIME>=0]
    med = med.sort_values('STARTTIME')
    medtimes = med.STARTTIME.unique()
    meds = np.empty((len(rnn_grid_times),len(medmap.keys())))

    return meds

def dataset_prep():
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    subject_indices = [x[1] for x in lines]
    cvmedseries = pd.read_csv('/data/MIMIC3/INPUTEVENTS_CV.csv', low_memory=False)
    cvmedseries = cvmedseries[cvmedseries['ORIGINALROUTE'].isin(['Intravenous', 'IV Drip', 'Drip'])]
    cvmedseries = cvmedseries[cvmedseries.SUBJECT_ID.isin(subject_ids)]
    nmeds = cvmedseries.ITEMID.unique()
    cvsubs = cvmedseries.SUBJECT_ID.unique()
    mvmedseries = pd.read_csv('/data/MIMIC3/INPUTEVENTS_MV.csv', low_memory=False)
    mvmedseries = mvmedseries[mvmedseries.ORDERCATEGORYDESCRIPTION=='Continuous IV']
    mvmedseries = mvmedseries[mvmedseries.SUBJECT_ID.isin(subject_ids)]
    mvsubs = mvmedseries.SUBJECT_ID.unique()
    nmeds = np.unique(np.concatenate((nmeds,mvmedseries.ITEMID.unique())))
    med_map = {n:i for i,n in enumerate(nmeds)}
    for index,sub in enumerate(subject_ids):
        stays = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/stays.csv', parse_dates=True)
        #timeline = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/episode1_timeseries.csv')
        intime = pd.to_datetime(stays['INTIME'])
        endtime = pd.to_datetime(stays['OUTTIME'])
        if sub in cvsubs:
            Sseries = getcvseries(cvmedseries[cvmedseries.SUBJECT_ID==sub], intime, endtime, med_map)
        elif sub in mvsubs:
            Sseries = getmvseries(mvmedseries[mvmedseries.SUBJECT_ID==sub], intime, endtime, med_map)
        #np.savetxt(str(sub)+'.med', Sseries, delimiter=',', newline='\n')
        meds = pd.DataFrame(data=Sseries)
        meds.to_csv(str(sub)+'.wav', index=False)


if __name__=="__main__":
    dataset_prep()
