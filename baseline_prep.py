import pandas as pd
import os
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
    gender_map = {}
    ethnicity_map = {}
    religion_map = {}
    language_map = {}
    marital_map = {}
    insurance_map = {}
    for sub in subject_ids:
        print("Processing subject:%s"%(str(sub)))
        person = admissions[admissions.SUBJECT_ID==sub]
        #print person
        insurance = person.INSURANCE.iloc[0]
        if insurance not in insurance_map.keys():
            insurance_map[insurance] = len(insurance_map)
        insurance = insurance_map[insurance]
        #print insurance_map
        language = person.LANGUAGE.iloc[0]
        if language not in language_map.keys():
            language_map[language] = len(language_map)
        language = language_map[language]
        religion = person.RELIGION.iloc[0]
        if religion not in religion_map.keys():
            religion_map[religion] = len(religion_map)
        religion = religion_map[religion]
        marital = person.MARITAL_STATUS.iloc[0]
        if marital not in marital_map.keys():
            marital_map[marital] = len(marital_map)
        marital = marital_map[marital]
        ethnicity = person.ETHNICITY.iloc[0]
        if ethnicity not in ethnicity_map.keys():
            ethnicity_map[ethnicity] = len(ethnicity_map)
        ethnicity = ethnicity_map[ethnicity]
        stays = pd.read_csv(dest_path+str(sub)+'/stays.csv')
        for stay_no in range(stays.shape[0]):
            try:
                epi = pd.read_csv(dest_path+str(sub)+'/episode'+str(stay_no+1)+'.csv')
            except:
                continue
            gender = epi.Gender.iloc[0]
            if gender not in gender_map.keys():
                gender_map[gender] = len(gender_map)
            gender = gender_map[gender]
            age = epi.Age.iloc[0]
            height = epi.Height.iloc[0]
            weight = epi.Weight.iloc[0]
            data = [gender, age, ethnicity, religion, language, height, weight, marital, insurance]
            #print data
            df = pd.DataFrame(data=[data], columns=cols)
            #print df
            df.to_csv(dest_path+str(sub)+'/baseline'+str(stay_no)+'.csv', index=False)

if __name__=="__main__":
    baselines()
