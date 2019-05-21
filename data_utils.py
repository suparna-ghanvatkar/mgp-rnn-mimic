import pandas as pd
import numpy as np
import os
import pickle
from math import ceil
import wfdb
import datetime
import random
from collections import defaultdict
import concurrent.futures
from shutil import copyfile

def count_numeric_signals():
    fwaves = open('/data/suparna/MatchedSubset_MIMIC3/RECORDS-numerics','r')
    waves = defaultdict(int)
    for line in fwaves:
        line = line.strip('\n')
        path = '/data/suparna/MatchedSubset_MIMIC3/'+line
        try:
            __,f = wfdb.rdsamp(path)
            for s in f['sig_name']:
                waves[s] += 1
            print "processed "+line
        except:
            print "ignore "+line
            continue
    print waves
    print sorted(waves, key=waves.get)

def list_substays():
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    waves = pickle.load(open('ecgII_index.pickle','r'))
    final_substays = []
    data_path = '/data/suparna/MGP_data/'
    for sub in subject_ids:
        stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
        #timeline = pd.read_csv('~/mimic3-benchmarks/data/root/'+str(subject_indices[index])+'/'+str(sub)+'/episode1_timeseries.csv')
        intime = pd.to_datetime(stays['INTIME'])
        outtime = pd.to_datetime(stays['OUTTIME'])
        starttime = intime.dt.round('1h')
        print("Processing %s"%sub)
        for stay_no in range(stays.shape[0]):
            try:
                timeline = pd.read_csv(data_path+'root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv')
            except:
                continue
            grid_times = list(np.arange(ceil(((outtime-starttime).dt.total_seconds()/(60*60))[stay_no])+1))
            if len(grid_times)<24:
                #cancelled_subs.append(sub)
                continue
            timeline = timeline[timeline.Hours>=0]
            timeline = timeline[timeline.Hours<=24]
            timeline = timeline.drop_duplicates()
            timeline = timeline.drop('Hours',axis=1)
            #tot = timeline.shape[0]*timeline.shape[1]
            nans = timeline.count().sum()
            #print nans
            Y_len = nans
            #if Y_len>200:
            #    continue
            #timeline = timeline.fillna(0)
            if timeline.empty:
                continue
            try:
                baseline = pd.read_csv(data_path+'root/'+str(sub)+'/baseline'+str(stay_no)+'.csv', )
            except:
                continue
            if waves[sub]==[]:
                continue
            wave_dates = waves[sub]
            for date,index in wave_dates:
                yr,MM,dd,hh,mm = map(int,date.split('-'))
                thisdate = datetime.datetime(yr,MM,dd,hh,mm)
                if thisdate>starttime[stay_no] and thisdate<(starttime[stay_no]+pd.Timedelta(hours=24)):
                    final_substays.append((sub,stay_no,date,index))
                    print(sub,stay_no)
                else:
                    continue
    pickle.dump(final_substays,open('final_substays.pickle','w'))

def get_signal_index():
    #this function is before generating final substays, the ecg signal indices are created
    fwaves = open('/data/suparna/MatchedSubset_MIMIC3/RECORDS-waveforms','r')
    waves = defaultdict(list)
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    match = open('/data/suparna/MatchedSubset_MIMIC3/p00/p000020/3544749_layout.hea','r').readlines()[1].split(' ')[-1].strip('\n')
    print match
    for line in fwaves:
        line = line.strip('\n')
        thispath = line.split('/')
        sub,date = thispath[2].split('-',1)
        sub = int(sub[1:])
        basepath = '/data/suparna/MatchedSubset_MIMIC3/'
        path = basepath+line+'.hea'
        lines = open(path,'r').readlines(2)
        layout_file = lines[1].split(' ')[0]
        path = basepath+thispath[0]+'/'+thispath[1]+'/'+layout_file+'.hea'
        fields = open(path,'r')
        #sig_i = -1
        lines = fields.readlines()
        for i in range(1,len(lines)):
            l = lines[i].strip('\n')
            print l.split(' ')[-1]
            if str(l.split(' ')[-1])==match:
                print "found in "+str(sub)
                if sub in subject_ids:
                    waves[sub].append((date,i-1))
                break
    pickle.dump(waves,open('ecgII_index.pickle','w'))

def get_add_signal_index():
    fwaves = open('/data/suparna/MatchedSubset_MIMIC3/RECORDS-waveforms','r')
    waves = {}
    #fsub = open('subject_presence','r')
    #lines = fsub.read().splitlines()
    #fsub.close()
    #lines = [(l.split()[0], l.split()[1]) for l in lines]
    #lines = [x for x in lines if x[1]!='absent']
    #subject_ids = [int(x[0]) for x in lines]
    sub_stay = pickle.load(open('final_substays.pickle','r'))
    sub_date = [(sub,date) for (sub,stay,date,ind) in sub_stay]
    #for ABP line number 3
    match = open('/data/suparna/MatchedSubset_MIMIC3/p00/p000020/3544749_layout.hea','r').readlines()[3].split(' ')[-1].strip('\n')
    print match
    for line in fwaves:
        line = line.strip('\n')
        thispath = line.split('/')
        sub,date = thispath[2].split('-',1)
        sub = int(sub[1:])
        basepath = '/data/suparna/MatchedSubset_MIMIC3/'
        path = basepath+line+'.hea'
        lines = open(path,'r').readlines(2)
        layout_file = lines[1].split(' ')[0]
        path = basepath+thispath[0]+'/'+thispath[1]+'/'+layout_file+'.hea'
        fields = open(path,'r')
        #sig_i = -1
        lines = fields.readlines()
        for i in range(1,len(lines)):
            l = lines[i].strip('\n')
            print l.split(' ')[-1]
            if str(l.split(' ')[-1])==match:
                print "found in "+str(sub)
                if (sub,date) in sub_date:
                    waves[(sub,date)] = i-1
                break
    pickle.dump(waves,open('abp_index.pickle','w'))

def create_bal_dataset():
    sub_stay = pickle.load(open('final_substays.pickle','r'))
    data_path = '/data/suparna/MGP_data/'
    #sub_stay = sub_stay[:9100]
    labels_pos = []
    labels_neg = []
    for sub,stay_no,date,index in sub_stay:
        stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv')
        label = stays['MORTALITY_INHOSPITAL'][stay_no]
        #print label
        if label==1:
            labels_pos.append((sub,stay_no,date,index))
        else:
            labels_neg.append((sub,stay_no,date,index))
    pickle.dump(labels_pos,open('positive_labels.pickle','w'))
    pickle.dump(labels_neg, open('negative_labels.pickle','w'))
    poscount = len(labels_pos)
    random.shuffle(labels_neg)
    #round off the total samples in data to be divisible by 5. For this, check if poscount is div by 5
    #else find the the number less for making it divisible by 5 and add twice this number to the len to negative lables to include
    if poscount%5==0:
        negcount = poscount
    else:
        negcount = int(2*((5*ceil(poscount/5.0))-poscount))+poscount
    labels_neg = labels_neg[:negcount]
    new_data = labels_pos+labels_neg
    random.shuffle(new_data)
    pickle.dump(new_data,open('balanced_data.pickle','w'))

def create_data_hierar():
    sub_stay = pickle.load(open('final_substays.pickle','r'))
    data_path = '/data/suparna/MGP_data/'
    #sub_stay = sub_stay[:9100]
    labels_pos = pickle.load(open('positive_labels.pickle','r'))
    labels_neg = pickle.load(open('negative_labels.pickle','r'))
    balanced_data = pickle.load(open('balanced_data.pickle','r'))
    #newcount = len(balanced_data)/2
    totcount = 2940
    newcount = totcount - len(balanced_data)
    #print totcount
    new_data = set(balanced_data)
    random.shuffle(labels_neg)
    labels_negs = labels_neg[:newcount+1]
    while not new_data.isdisjoint(labels_negs) and len(new_data)<totcount:
        print len(new_data),len(labels_negs)
        new_data = set(new_data).union(labels_negs)
        random.shuffle(labels_neg)
        labels_negs = labels_neg[:totcount-len(new_data)]
    new_data = set(new_data).union(labels_negs)
    print len(new_data)
    #new_data = balanced_data + labels_neg
    balanced_data_test_1 = pickle.load(open('balanced_data_test_1.pickle','r'))
    #balanced_data_train_1 = pickle.load(open('balanced_data_train_0.pickle','r'))
    balanced_data_train_1 = list(new_data.difference(balanced_data_test_1))
    pickle.dump(balanced_data_train_1,open('hierar_data_train_1.pickle','w'))
    pickle.dump(balanced_data_test_1,open('hierar_data_test_1.pickle','w'))

def train_test():
    #split the balanced data pickle obtained into train test splits 5 folds...
    #sub_stay = pickle.load(open('balanced_data.pickle','r'))
    sub_stay = pickle.load(open('hierar_data.pickle','r'))
    count = len(sub_stay)
    for i in range(5):
        #80-20 split
        #N_te = int(0.2*count)
        N_te = 390
        N_tr = count-N_te
        random.shuffle(sub_stay)
        tr = sub_stay[:N_tr]
        te = sub_stay[N_tr:]
        pickle.dump(tr,open('hierar_data_train_'+str(i)+'.pickle','w'))
        pickle.dump(te,open('hierar_data_test_'+str(i)+'.pickle','w'))

def create_baseline_input():
    #create the file [stay, y_true] csv file for the benchmark in-hosp mortality
    #first create the train and test folders in in-hosp benchmark folder - do this in the shell itself before running the function
    for i in range(5):
        train = pickle.load(open('balanced_data_train_'+str(i)+'.pickle','r'))
        test = pickle.load(open('balanced_data_test_'+str(i)+'.pickle','r'))
        #find each episode file and copy it into the folds train or test folder
        trainfile = open('/data/suparna/MGP_data/lstm/'+str(i)+'/train_listfile.csv','w')
        trainfile.write("stay,y_true\n")
        testfile = open('/data/suparna/MGP_data/lstm/'+str(i)+'/val_listfile.csv','w')
        testfile.write("stay,y_true\n")
        fold_dir = '/data/suparna/MGP_data/lstm/'+str(i)+'/'
        data_path = '/data/suparna/MGP_data/'
        for sub,stay_no,date,index in train:
            stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
            age = stays['AGE'][stay_no]
            if age<18:
                print("age of sub %s is %s"%(sub,age))
            label = stays['MORTALITY_INHOSPITAL'][stay_no]
            src_path = '/data/suparna/MGP_data/root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv'
            fname = str(sub)+'_episode'+str(stay_no)+'_timeseries.csv'
            dest_path = fold_dir+'train/'+fname
            timeline = pd.read_csv(src_path)
            timeline = timeline[['Hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response','Glucose','Heart Rate','Height','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure','Temperature','Weight','pH']]
            timeline = timeline[timeline.Hours>=0]
            timeline = timeline[timeline.Hours<=24]
            timeline = timeline.drop_duplicates()
            timeline.to_csv(dest_path,index=False)
            #copyfile(src_path,dest_path)
            trainfile.write("%s,%s\n"%(fname,str(label)))
        for sub,stay_no,date,index in test:
            stays = pd.read_csv(data_path+'root/'+str(sub)+'/stays.csv', parse_dates=True)
            age = stays['AGE'][stay_no]
            if age<18:
                print("age of sub %s is %s"%(sub,age))
            label = stays['MORTALITY_INHOSPITAL'][stay_no]
            src_path = '/data/suparna/MGP_data/root/'+str(sub)+'/episode'+str(stay_no+1)+'_timeseries.csv'
            timeline = pd.read_csv(src_path)
            timeline = timeline[['Hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response','Glucose','Heart Rate','Height','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure','Temperature','Weight','pH']]
            timeline = timeline[timeline.Hours>=0]
            timeline = timeline[timeline.Hours<=24]
            timeline = timeline.drop_duplicates()
            fname = str(sub)+'_episode'+str(stay_no)+'_timeseries.csv'
            dest_path = fold_dir+'test/'+fname
            #copyfile(src_path,dest_path)
            timeline.to_csv(dest_path,index=False)
            testfile.write("%s,%s\n"%(fname,str(label)))
        trainfile.close()
        testfile.close()





def split_into_5_fold():
    sub_stay = pickle.load(open('final_substays.pickle','r'))
    sub_stay = sub_stay[:9100]
    count = len(sub_stay)
    for i in range(5):
        #80-20 split for 9106 is 7284 and 1822
        #train: 5462, val and tes: 1822 each
        N_te = int(0.2*count)
        N_val = N_te
        N_tr = count-N_te-N_val
        random.shuffle(sub_stay)
        tr = sub_stay[:N_tr]
        val = sub_stay[N_tr:N_tr+N_val]
        te = sub_stay[-N_te:]
        pickle.dump(tr,open('final_substays_train_'+str(i)+'.pickle','w'))
        pickle.dump(te,open('final_substays_test_'+str(i)+'.pickle','w'))
        pickle.dump(val,open('final_substays_val_'+str(i)+'.pickle','w'))

if __name__=="__main__":
    #count_numeric_signals()
    #get_signal_index()
    #list_substays()
    #split_into_5_fold()
    #create_bal_dataset()
    #train_test()
    #create_data_hierar()
    #create_baseline_input()
    get_add_signal_index()

