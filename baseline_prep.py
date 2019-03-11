import pandas as pd
import numpy as np
from math import ceil, isnan, floor
#script to generate baseline demographics of patient of MIMIC3 - 9 demographics
def baselines():
    mimic_path = '/data/MIMIC3/'
    admissions = pd.read_csv(os.path.join(mimic_path,'ADMISSIONS.csv'))
    dest_path = '/data/suparna/MGP_data/root/'
    fsub = open('subject_presence','r')
    lines = fsub.read().splitlines()
    fsub.close()
    lines = [(l.split()[0], l.split()[1]) for l in lines]
    lines = [x for x in lines if x[1]!='absent']
    subject_ids = [int(x[0]) for x in lines]
    cols = ["Gender", "Age", "Ethnicity", "Religion", "Language", "Height", "Weight", "Marital_Status", "Insurance"]
    for sub in subject_ids:
        person = admissions[admissions.SUBJECT_ID==sub]
        insurance = person.INSURANCE[0]
        language = person.LANGUAGE[0]
        religion = person.RELIGION[0]
        marital = person.MARITAL_STATUS[0]
        ethnicity = person.ETHNICITY[0]
        stays = pd.read_csv(dest_path+str(sub)+'/stays.csv')
        for stay_no in range(stays.shape[0]):
            epi = pd.read_csv(dest_path+str(sub)+'/episode'+stay_no+'.csv')
            gender = epi.GENDER[0]
            age = epi.AGE[0]
            height = epi.HEIGHT[0]
            weight = epi.WEIGHT[0]
            data = [gender, age, ethnicity, religion, language, height, weight, marital, insurance]
            pd.DataFrame(data=[data], columns=cols).to_csv(dest_path+str(sub)+'/baseline'+str(stay_no)+'.csv')
            
    