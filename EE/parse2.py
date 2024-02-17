import json

import pandas as pd

from tqdm import tqdm
import os
from string import punctuation
import re

stat_dict = {
    'not_specified': 0,
    'same_with_level1': 0,
    'unmatched_relation': 0,
    'level2_events': 0,
    'object_lost': 0
}

filtered_records = {
    'not_specified': {}, # rowid:response
    'same_with_level1': {},
    'unmatched_relation': {},
    'unmatched_format' : {}
}

MANUAL = {
    "Engage in economic cooperation": "Cooperate economically",
    "Engage in military cooperation": "Cooperate militarily",
    "Express optimistic comment": "Make optimistic comment",
    "Cooperate judicially": "Engage in judicial cooperation",
    "Demand for material cooperation": "Demand material cooperation",
    "Demand for material aid": "Demand material aid",
    "Demand for political reform": "Demand political reform",
    "Demand for meeting, negotiation": "Demand meeting, negotiation",
    "Demand for settling of dispute": "Demand settling of dispute",
    "Demand for mediation": "Demand mediation",
    "Investigate corruption": "Investigate crime, corruption",
    "Investigate crime": "Investigate crime, corruption",
    "Express intent to engage in political reform": "Express intent to institute political reform",
    "Fight with aerial weapons": "Employ aerial weapons",
    "Demand for a cease-fire": "Demand that target yield or concede",
    "Express intent to engage in military cooperation": "Express intent to engage in material cooperation",
    "Appeal for military cooperation": "Appeal for material cooperation",
    "Appeal for judicial cooperation": "Appeal for material cooperation",
}

MANUAL_PATTERNS = {
    "Demand .* release .*": "Demand that target yield or concede"
}

def main(level1_event_df, exists_results, dict_id2ont, dict_hier_id):
    # build level 2 relation name2id dict (id is str)
    level2_name2id = {}
    level2_ids = []
    for level1, info in dict_hier_id.items():
        level2_ids += list(info.keys())
    for level2_id in level2_ids:
        level2_name2id[dict_id2ont[level2_id]['choice'].lower()] = level2_id
    level2_name2id['not specified']  = '000'
    print('Level 2 relation name2id dict is biult.')

    all_events = []

    for idx, rowid in tqdm(enumerate(exists_results), total=len(exists_results)):
        level1_row = level1_event_df.iloc[[rowid]]

        s = level1_row['Subject'].item()
        level1_r_choice = level1_row['Relation_choice'].item()
        o = level1_row['Object'].item()
        md5 = level1_row['Md5'].item()
        id = level1_row['ID'].item()
        trunk = level1_row['Trunk'].item()

        with open('results_v2/raw_results/level2/ ' + id, 'r') as fraw:
            rsp = fraw.read()

            level2_name = ''

            for key, value in level2_name2id.items():
                if key in rsp.lower():
                    level2_name = key
                    break

            if level2_name == '':
                stat_dict['unmatched_relation'] += 1
                filtered_records['unmatched_relation'][id] = rsp
                continue

            if level2_name == 'not specified':
                stat_dict['not_specified'] += 1
                filtered_records['not_specified'][id] = rsp
                continue

            # check if is the same with level 1 relation
            if level2_name == level1_r_choice.lower():
                stat_dict['same_with_level1'] += 1
                filtered_records['same_with_level1'][rowid] = rsp
                continue

            level2_r_id = level2_name2id[level2_name]
            level2_r_choice = dict_id2ont[level2_r_id]['choice']

            stat_dict['level2_events'] += 1
            all_events.append([s, level2_r_id, level2_r_choice, o, md5, trunk, id])

    all_events_df = pd.DataFrame(all_events,
                        columns=['Subject', 'Relation_id', 'Relation_choice', 'Object', 'Md5', 'Trunk', 'ID'],
                        dtype='string')

    print(stat_dict)

    all_events_df.to_csv(path_or_buf='results_v2' + '/clean_results/level2/' + 'all_events.csv',
                         sep='\t', index=False)
    json.dump(stat_dict, open('results_v2' + '/clean_results/level2/' + 'stats.json', 'w'), indent=4)
    json.dump(filtered_records, open('results_v2' + '/clean_results/level2/' + 'filtered.json', 'w'), indent=4)


if __name__ == "__main__":
    # load level 1 events
    version = 'v2'
    level1_event_df = pd.read_csv('results_' + version + '/clean_results/level1/' + 'all_events.csv', sep='\t', dtype='string')

    # check results
    exists_result = os.listdir('results_v2/raw_results/level2')
    exists_results = [file for file in exists_result if file != '.DS_Store']

    dict_id2ont = json.load(open('dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('dict_hier_id.json', 'r'))

    main(level1_event_df, exists_results, dict_id2ont, dict_hier_id)
