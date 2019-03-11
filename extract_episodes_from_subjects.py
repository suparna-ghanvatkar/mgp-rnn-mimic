from __future__ import absolute_import
from __future__ import print_function

import argparse
import pandas as pd

import os
import sys

from subject import read_stays, read_diagnoses, read_events, get_events_for_stay, add_hours_elpased_to_events
from subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, read_variable_ranges, clean_events
from preprocessing import transform_gender, transform_ethnicity, assemble_episodic_data


parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
args, _ = parser.parse_known_args()

var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.VARIABLE.unique()

for subject_dir in os.listdir(args.subjects_root_path):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue
    sys.stdout.write('Subject {}: '.format(subject_id))
    sys.stdout.flush()

    try:
        sys.stdout.write('reading...')
        sys.stdout.flush()
        path = os.path.join(args.subjects_root_path, subject_dir)
        stays = pd.read_csv(os.path.join(path,'stays.csv'))
        diagnoses = pd.read_csv(os.path.join(path,'diagnoses.csv'))
        events = pd.read_csv(os.path.join(path,'events.csv'))
        stays = read_stays(stays)
        #diagnoses = read_diagnoses(diagnoses)
        events = read_events(events)
    except:
        sys.stdout.write('error reading from disk!\n')
        #path = (os.path.join(os.path.join(args.subjects_root_path, subject_dir), 'events.csv'))
        #df=pd.read_csv(path)
        #print(path)
        #print(df.shape)
        continue
    else:
        sys.stdout.write('got {0} stays, {1} diagnoses, {2} events...'.format(stays.shape[0], diagnoses.shape[0], events.shape[0]))
        sys.stdout.flush()

    try:
        episodic_data = assemble_episodic_data(stays, diagnoses)
    except:
        continue

    sys.stdout.write('cleaning and converting to time series...')
    sys.stdout.flush()
    events = map_itemids_to_variables(events, var_map)
    events = clean_events(events)
    if events.shape[0] == 0:
        sys.stdout.write('no valid events!\n')
        continue
    timeseries = convert_events_to_timeseries(events, variables=variables)

    sys.stdout.write('extracting separate episodes...')
    sys.stdout.flush()

    for i in range(stays.shape[0]):
        stay_id = stays.ICUSTAY_ID.iloc[i]
        sys.stdout.write(' {}'.format(stay_id))
        sys.stdout.flush()
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            sys.stdout.write(' (no data!)')
            sys.stdout.flush()
            continue

        episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        episodic_data.Weight.ix[stay_id] = get_first_valid_from_timeseries(episode, 'Weight')
        episodic_data.Height.ix[stay_id] = get_first_valid_from_timeseries(episode, 'Height')
        episodic_data.ix[episodic_data.index==stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}.csv'.format(i+1)), index_label='Icustay')
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)), index_label='Hours')
    sys.stdout.write(' DONE!\n')
