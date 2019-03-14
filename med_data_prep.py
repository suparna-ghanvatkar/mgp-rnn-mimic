import pandas as pd
import numpy as np
from math import ceil, isnan, floor

def isnat(dt):
    dtyp = str(dt.dtype)
    if 'datetime64' in dtyp or 'timedelta64' in dtyp:
        return dt.view('i8')==np.datetime64('NAT').view('i8')
    return False

def getcvseries(med, intime, outtime, medmap, stay_no):
    #print intime[stay_no]
    #print outtime[stay_no]
    #if isnat(intime[stay_no]) and isnat(outtime[stay_no]):
    #    return -1
    try:
        #print("time info")
        #print inttime[stay_no], outtime[stay_no]
        starttime = intime.dt.round('1h')
        rnn_grid_times = np.arange(ceil(((outtime-starttime).dt.total_seconds()/(60*60))[stay_no])+1)
        #print len(rnn_grid_times)
    except:
        return -1
    print "time info"
    print intime[stay_no], outtime[stay_no]
    print len(rnn_grid_times)
    endlen = len(rnn_grid_times)
    medtimes = pd.to_datetime(med.CHARTTIME)
    medt = medtimes.apply(lambda x: x-starttime)[stay_no]
    medt = medt.dt.total_seconds()/3600
    med.CHARTTIME = medt
    #medications have to be within the stay
    med = med[med.CHARTTIME>=0]
    med = med[med.CHARTTIME<endlen]
    med = med.sort_values('CHARTTIME')
    #medtimes = med.CHARTTIME.unique()
    meds = np.zeros((len(rnn_grid_times),len(medmap.keys())))
    for m,v in medmap.iteritems():
        medseries = med[med.ITEMID==m]
        for row in medseries.itertuples():
            #the rows either have amounts or else have rates
            amount = getattr(row, 'AMOUNT')
            rate = getattr(row, 'RATE')
            charttime = getattr(row, 'CHARTTIME')
            if not isnan(amount):
                #not if 14.5 is charttime, the it was given between 13-14 hr and also between 14-15 hr...but as rate not present, just add to end of 15th hour
                prevhr = row.charttime-floor(charttime)
                thishr = 1.0-prevhr
                meds[floor(charttime), v] += (prevhr*amount)
                meds[ceil(charttime), v] += (thishr*amount)
            elif not isnan(rate):
                rateuom = getattr(row,'RATEUOM')
                try:
                    if 'min' in rateuom:
                        #convert to per hours and consider:
                        rate = rate*60.0
                    else:
                        rate = rate*1.0
                    thishr = ceil(charttime)-charttime
                    meds[int(ceil(charttime)), v] += (thishr*rate)
                    meds[int(ceil(charttime))+1:, v] += rate
                except:
                    continue
            else:
                #both are nan skip
                continue
    return meds

def getmvseries(med, intime, outtime, medmap, stay_no):
    #if isnat(intime[stay_no]) and isnat(outtime[stay_no]):
    #    return -1
    try:
        starttime = intime.dt.round('1h')
        rnn_grid_times = np.arange(ceil(((outtime-starttime).dt.total_seconds()/(60*60))[stay_no])+1)
    except:
        return -1
    print "time info"
    print intime[stay_no], outtime[stay_no]
    print len(rnn_grid_times)
    endlen = len(rnn_grid_times)
    medstimes = pd.to_datetime(med.STARTTIME)
    medetimes = pd.to_datetime(med.ENDTIME)
    medt = medstimes.apply(lambda x: x-starttime)[stay_no]
    medt = medt.dt.total_seconds()/3600
    med.STARTTIME = medt
    #the start time of medications has to be within the stay
    med = med[med.STARTTIME>=0]
    med = med[med.STARTTIME<endlen]
    med = med.sort_values('STARTTIME')
    medt = medetimes.apply(lambda x: x-starttime)[stay_no]
    medt = medt.dt.total_seconds()/3600
    med.ENDTIME = medt
    med = med[med.ENDTIME>=0]
    #medtimes = med.CHARTTIME.unique()
    meds = np.zeros((len(rnn_grid_times),len(medmap.keys())))
    for m,v in medmap.iteritems():
        medseries = med[med.ITEMID==m]
        for row in medseries.itertuples():
            rate = getattr(row, 'RATE')
            starttime = getattr(row, 'STARTTIME')
            endtime = getattr(row, 'ENDTIME')
            rateuom = getattr(row,'RATEUOM')
            try:
                if 'min' in rateuom:
                    #convert to per hours and consider:
                    rate = rate*60.0
                else:
                    rate = rate*1.0
                thishr = ceil(starttime)-starttime
                meds[int(ceil(starttime)), v] += (thishr*rate)
                meds[int(ceil(starttime))+1:int(floor(endtime)), v] += rate
                lasthr = endtime - floor(endtime)
                meds[int(ceil(endtime)), v] += (lasthr*rate)
            except:
                continue
    return meds


def dataset_prep():
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    #subject_indices = [x[1] for x in lines]
    print("Reading data...")
    cvmedseries = pd.read_csv('/data/MIMIC3/INPUTEVENTS_CV.csv', low_memory=False)
    cvmedseries = cvmedseries[cvmedseries['ORIGINALROUTE'].isin(['Intravenous', 'IV Drip', 'Drip'])]
    cvmedseries = cvmedseries[cvmedseries.SUBJECT_ID.isin(subject_ids)]
    nmeds = cvmedseries.ITEMID.unique()
    cvsubs = cvmedseries.SUBJECT_ID.unique()
    mvmedseries = pd.read_csv('/data/MIMIC3/INPUTEVENTS_MV.csv', low_memory=False)
    mvmedseries = mvmedseries[mvmedseries.ORDERCATEGORYDESCRIPTION=='Continuous IV']
    mvmedseries = mvmedseries[mvmedseries.SUBJECT_ID.isin(subject_ids)]
    print("Reading data done")
    #the rewritten entries are incorrect and hence removed
    mvmedseries = mvmedseries[mvmedseries.STATUSDESCRIPTION!='Rewritten']
    mvsubs = mvmedseries.SUBJECT_ID.unique()
    #nmeds = np.unique(np.concatenate((nmeds,mvmedseries.ITEMID.unique())))
    nmeds = {225158, 30131, 30045, 30025, 225166}
    med_map = {n:i for i,n in enumerate(nmeds)}
    for index,sub in enumerate(subject_ids):
        stays = pd.read_csv('/data/suparna/MGP_data/root/'+str(sub)+'/stays.csv', parse_dates=True)
        print "subject "+str(sub)
        #timeline = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/episode1_timeseries.csv')
        intime = pd.to_datetime(stays['INTIME'])
        outtime = pd.to_datetime(stays['OUTTIME'])
        #starttime = intime.dt.round('1h')
        for s in range(stays.shape[0]):
            print("stay:%s"%(s))
            if sub in cvsubs:
                Sseries = getcvseries(cvmedseries[cvmedseries.SUBJECT_ID==sub], intime, outtime, med_map, s)
            elif sub in mvsubs:
                Sseries = getmvseries(mvmedseries[mvmedseries.SUBJECT_ID==sub], intime, outtime, med_map, s)
            else:
                continue
            if isinstance(Sseries,(int, long)):
                continue
            #np.savetxt(str(sub)+'.med', Sseries, delimiter=',', newline='\n')
            meds = pd.DataFrame(data=Sseries, columns=med_map.keys())
            print("writing to csv")
            meds.to_csv('/data/suparna/MGP_data/medicines/'+str(sub)+'_stay'+str(s)+'.med', index=False)


if __name__=="__main__":
    dataset_prep()
