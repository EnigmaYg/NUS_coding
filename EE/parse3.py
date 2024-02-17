import json

import pandas as pd

from tqdm import tqdm
import os
from string import punctuation
import re

stat_dict = {
    'not_specified': 0,
    'same_with_level2': 0,
    'unmatched_relation': 0,
    'level3_events': 0,
    'no_level3': 0,
}

filtered_records = {
    'not_specified': {}, # rowid:response
    'same_with_level2': {},
    'unmatched_relation': {},
    'no_level3': {}
}

MANUAL = {
    "Appeal for financial support": "Appeal for economic cooperation"
}

MANUAL_PATTERNS = {
}

def main(level2_event_df, exists_results, dict_id2ont, dict_hier_id):
    # build level 2 relation name2id dict (id is str)
    level3_name2id = {}
    level3_ids = []
    for level2, level3s in dict_hier_id.items():
        level3_ids += level3s
    for level3_id in level3_ids:
        level3_name2id[dict_id2ont[level3_id]['choice'].lower()] = level3_id
    level3_name2id['not specified'] = '000'
    print('Level 3 relation name2id dict is biult.')

    all_events = []

    for idx, rowid in tqdm(enumerate(exists_results), total=len(exists_results)):
        try:
            level2_row = level2_event_df.iloc[[idx]]
        except:
            continue
        level2_rid = level2_row['Relation_choice'].item()
        if level2_rid in dict_hier_id:
            print(rowid)
        s = level2_row['Subject'].item()
        level2_r_choice = level2_row['Relation_choice'].item()
        o = level2_row['Object'].item()
        md5 = level2_row['Md5'].item()
        id = level2_row['ID'].item()
        trunk = level2_row['Trunk'].item()

        with open('results_v2/raw_results/level3/' + rowid, 'r') as fraw:
            rsp = fraw.read()
            level3_no = 'no level3 event'
            level3_name = ''

            if level3_no in rsp.lower():
                stat_dict['no_level3'] += 1
                filtered_records['no_level3'][rowid] = rsp
                continue

            for key, value in level3_name2id.items():
                if key in rsp.lower():
                    level3_name = key
                    break

            if level3_name == '':
                stat_dict['unmatched_relation'] += 1
                filtered_records['unmatched_relation'][id] = rsp
                continue

            if level3_name == 'not specified':
                stat_dict['not_specified'] += 1
                filtered_records['not_specified'][id] = rsp
                continue

            # check if is the same with level 1 relation
            if level3_name == level2_r_choice.lower():
                stat_dict['same_with_level2'] += 1
                filtered_records['same_with_level2'][rowid] = rsp
                continue

            level3_r_id = level3_name2id[level3_name]
            level3_r_choice = dict_id2ont[level3_r_id]['choice']

            stat_dict['level3_events'] += 1
            all_events.append([s, level3_r_id, level3_r_choice, o, md5, trunk, id])

    all_events_df = pd.DataFrame(all_events,
                                 columns=['Subject', 'Relation_id', 'Relation_choice', 'Object', 'Md5', 'Trunk', 'ID'],
                                 dtype='string')

    print(stat_dict)

    all_events_df.to_csv(path_or_buf='results_v2' + '/clean_results/level3/' + 'all_events.csv',
                         sep='\t', index=False)
    json.dump(stat_dict, open('results_v2' + '/clean_results/level3/' + 'stats.json', 'w'), indent=4)
    json.dump(filtered_records, open('results_v2' + '/clean_results/level3/' + 'filtered.json', 'w'), indent=4)


if __name__ == "__main__":
    version = 'v2'
    level2_event_df = pd.read_csv('results_' + version + '/clean_results/level2/' + 'all__events.csv', sep='\t',
                                  dtype='string')

    # check results
    exists_result = os.listdir('results_v2/raw_results/level3')
    exists_results = [file for file in exists_result if file != '.DS_Store']
    print(len(exists_results))

    dict_id2ont = json.load(open('dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('dict_hier_id_level23.json', 'r'))

    main(level2_event_df, exists_results, dict_id2ont, dict_hier_id)
