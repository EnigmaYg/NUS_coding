import json

import pandas as pd

from tqdm import tqdm
import os
from string import punctuation

stat_dict = {
    'no_event': 0,
    'invalid_format': 0,
    'listed_triplets': 0,
    'unmatched_relation': 0,
    'unknown_actor': 0,
    'level1_events': 0,  # after check relation and actor
}

filtered_records = {
    'no_event': {}, # md5: rsp
    'invalid_format': {}, # md5: [events]
    'unmatched_relation': {},
    'unknown_actor': {}
}

UNKNOWN_ACTOR = ['unknown', 'none']

def main(exists_results, dict_id2ont, dict_hier_id):
    # build level 1 relation name2id dict (id is str)
    level1_name2id = {}
    level1_ids = list(dict_hier_id.keys())
    for level1_id in level1_ids:
        level1_name2id[dict_id2ont[level1_id]['choice'].lower()] = level1_id
        level1_name2id[dict_id2ont[level1_id]['name'].lower()] = level1_id
    print('Level 1 relation name2id dict is biult.')

    all_events = []

    for idx, md5 in tqdm(enumerate(exists_results), total=len(exists_results)):

        with open('raw_results/results/' + md5, 'r') as fraw:
            rsp = fraw.read()
            rsp = rsp.strip(' |\n' + punctuation)

            # check if no structured event is extracted
            if ';' not in rsp:
                stat_dict['no_event'] += 1
                filtered_records['no_event'][md5] = rsp
                continue

            # check each event
            events = rsp.split('|')
            for event in events:
                # check format
                fields = event.split(';')
                if len(fields) != 3:
                    stat_dict['invalid_format'] += 1
                    if md5 not in filtered_records['invalid_format']:
                        filtered_records['invalid_format'][md5] = []
                    filtered_records['invalid_format'][md5].append(event)
                    continue
                s, r, o = [_.strip(' ') for _ in fields]
                stat_dict['listed_triplets'] += 1

                # check relation
                if r.lower() not in level1_name2id:
                    stat_dict['unmatched_relation'] += 1
                    if md5 not in filtered_records['unmatched_relation']:
                        filtered_records['unmatched_relation'][md5] = []
                    filtered_records['unmatched_relation'][md5].append(event)
                    continue
                r_id = level1_name2id[r.lower()]
                r_choice = dict_id2ont[r_id]['choice']

                # check actor
                if s.lower() in UNKNOWN_ACTOR or o.lower() in UNKNOWN_ACTOR:
                    stat_dict['unknown_actor'] += 1
                    if md5 not in filtered_records['unknown_actor']:
                        filtered_records['unknown_actor'][md5] = []
                    filtered_records['unknown_actor'][md5].append(event)
                    continue
                stat_dict['level1_events'] += 1
                all_events.append([s, r_id, r_choice, o, md5])

    all_events_df = pd.DataFrame(all_events,
                        columns=['Subject', 'Relation_id', 'Relation_choice', 'Object', 'Md5'],
                        dtype='string')

    print(stat_dict)

    version = 'v1'
    if not os.path.isdir('results_' + version + '/level1'):
        print('Making new dir: ' + 'results_v1/level1')
        os.makedirs('results_v1/level1')

    all_events_df.to_csv(path_or_buf = 'results_' + version + '/level1/' + 'events.csv',
                         sep='\t', index=False)
    json.dump(stat_dict, open('results_' + version + '/level1/' + 'stats.json', 'w'), indent=4)
    json.dump(filtered_records, open('results_' + version + '/level1/' + '_filtered.json', 'w'), indent=4)


if __name__ == "__main__":
    mread = json.load(open('md5list.json', 'r'))
    md5_list = mread['md5']

    exists_results = os.listdir('raw_results/results')
    # exists_results.remove('.DS_Store')

    dict_id2ont = json.load(open('dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('dict_hier_id.json', 'r'))
    main(exists_results, dict_id2ont, dict_hier_id)