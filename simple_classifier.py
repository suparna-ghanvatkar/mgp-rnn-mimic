import pandas as pd
import numpy as np
from math import ceil, isnan
import pickle
from collections import defaultdict
import wfdb
from time import time
import datetime
#from sklearn.preprocessing import StandardScaler, Imputer, scale
import concurrent.futures
import argparse
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
'''
The baselines are: Ethnicity, Gender, Age, Height, Weight
'''
glascow_eye_open = {'To Pain':0,'3 To speech':1,'1 No Response':2,'4 Spontaneously':3,'None':4,'To Speech':5,'Spontaneously':6,'2 To pain':7}
glascow_motor = {'1 No Response':0,'3 Abnorm flexion':1,'Abnormal extension':2,'No response':3,'4 Flex-withdraws':4,'Localizes Pain':5,'Flex-withdraws':6,'Obeys Commands':7,'Abnormal Flexion':8,'6 Obeys Commands':9,'5 Localizes Pain':10,'2 Abnorm extensn':11}
glascow_verbal = {'1 No Response':0,'No Response':1,'Confused':2,'Inappropriate Words':3,'Oriented':4,'No Response-ETT':5,'5 Oriented':6,'Incomprehensible sounds':7,'1.0 ET/Trach':8,'4 Confused':9,'2 Incomp sounds':10,'3 Inapprop words':11}


data_path = '/data/suparna/MGP_data/'


'''
Each feature used in the dataset, i.e. the 25 labs, 5 medicines and 1 waveform - find the descriptive stats like
max min mean median mode std var range kurtosis skewness avg power and energy spectral density
the 5 baselines are added as it is.
for categorical data like glascow scales, use count of each category as a feature
'''
Ethnicity = {}
eth_count = 0
Gender = {'M':0, 'F':1}
#for generating the discrete numeric lables for glascow columns
def get_encounter(values):
    (sub,stay_no),data_index = values
    sub = int(sub)
    stay_no = int(stay_no)
    feats = []
    #print("now subject"+str(sub))
    stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
    intime = pd.to_datetime(stays['INTIME'])
    outtime = pd.to_datetime(stays['OUTTIME'])
    starttime = intime.dt.round('1h')
    label = stays['MORTALITY_INHOSPITAL'][stay_no]
    timeline = pd.read_csv(data_path+'root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv')
    grid_times = range(24)
    timeline = timeline[timeline.Hours>=0]
    timeline = timeline[timeline.Hours<=24]
    timeline = timeline.drop_duplicates()
    substr = "%06d"%(sub)
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
    waveform = np.column_stack((np.mean(waveform.reshape(-1,125), axis=1), np.std(waveform.reshape(-1,125), axis=1)))
    signal = np.nan_to_num(waveform)
    baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv', )
    col_del = ['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale verbal response']
    t = timeline.drop(col_del,axis=1)
    #print t.columns
    description = t.describe()
    #remove Hours column from this
    #seq is count, mean std min median max
    #description = description.drop('Hours', axis=1)
    for col in t.columns:
        try:
            vals = description[col].drop(['25%','75%'],axis=0)
            vals = list(vals)
            #print col+str(len(vals))
            feats.extend(vals)
        except:
            feats.extend([0]*6)
    #now the counts of glascow scales:
    #now the waveform
    s = signal.ravel()
    sig_feats = [np.count_nonzero(~np.isnan(s)), np.nanmean(s), np.nanstd(s), np.nanmin(s), np.nanmedian(s), np.nanmax(s)]
    feats.extend(sig_feats)
    #print len(sig_feats)
    #now medicines
    try:
        medicines = pd.read_csv(data_path+'medicines/'+str(sub)+'_stay'+str(stay_no)+'.med',nrows=24)
        description = medicines.describe()
        for col in description.columns:
            vals = description[col].drop(['25%','75%'],axis=0)
            vals = list(vals)
            feats.extend(vals)
        #print("adding meds:%s"%len(feats))
    except:
        #print("no meds")
        feats.extend([0]*(5*6))
    baseline = baseline.fillna(0)
    baseline_i = baseline.iloc[0].to_list()
    feats.extend(baseline_i)
    #print len(feats)
    return feats,label

def data_prep(train,fold):
    sub_stay = pickle.load(open('icis_revision/filtered_substays_'+train+'_fold'+str(fold)+'.pickle','rb'))
    #sub_stay = sub_stay[:30]
    #print sub_stay
    count = len(sub_stay)
    dataset = []
    labels = []
    '''
    for value in sub_stay:
        sample,lab = get_encounter(value)
        dataset.append(sample)
        labels.append(lab)
    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #for sub,stay_no,date,index in sub_stay:
        results = executor.map(get_encounter, sub_stay,chunksize=3)
        for sample,lab in results:
            dataset.append(list(np.nan_to_num(np.array(sample))))
            labels.append(lab)
    #'''
    return dataset,labels

if __name__=="__main__":
    #data_prep('train',0)
    parser = argparse.ArgumentParser()
    parser.add_argument('fold', type=int, help='fold_number')
    args = parser.parse_args()

    x_train,y_train = data_prep('train',args.fold)
    #x_train = np.nan_to_num(np.array(x_train))
    #print x_train
    #print y_train
    x_test,y_test = data_prep('test',args.fold)
    #x_test = np.nan_to_num(np.array(x_test))
    pickle.dump(y_test, open('icis_revision/simpleclassifier_targ_fold'+str(args.fold)+'.pickle','wb'))
    print("For KNN")
    neigh = KNeighborsClassifier(n_neighbors=60, weights='distance')
    neigh.fit(x_train,y_train)
    y_pred = neigh.predict(x_test)
    print(roc_auc_score(y_test,y_pred))
    pickle.dump(y_pred, open('icis_revision/knn_labels_fold'+str(args.fold)+'.pickle','wb'))
    #'''
    print("For SVM")
    clf_svm = svm.SVC(kernel='sigmoid')
    clf_svm.fit(x_train,y_train)
    y_pred = clf_svm.predict(x_test)
    print(roc_auc_score(y_test,y_pred))
    pickle.dump(y_pred, open('icis_revision/svm_labels_fold'+str(args.fold)+'.pickle','wb'))
    #'''

    print("For MLP")
    clf_mlp = MLPClassifier(hidden_layer_sizes=(200,),solver='adam')
    clf_mlp.fit(x_train,y_train)
    y_pred = clf_mlp.predict(x_test)
    pickle.dump(y_pred, open('icis_revision/mlp_labels_fold'+str(args.fold)+'.pickle','wb'))
    print(roc_auc_score(y_test,y_pred))
