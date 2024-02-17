import os
import json
import time
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

id_cnt = 0

if __name__ == '__main__':
    exists_results = os.listdir('results_v2/raw_results/level11/')

    dict_id2ont = json.load(open('dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('dict_hier_id.json', 'r'))
    trunk_id = json.load(open('results_v2/trunk_id_.json', 'r'))
    level1_name2id = {}
    level1_ids = list(dict_hier_id.keys())
    for level1_id in level1_ids:
        level1_name2id[dict_id2ont[level1_id]['choice'].lower()] = level1_id
        level1_name2id[dict_id2ont[level1_id]['name'].lower()] = level1_id
    all_event = {}
    for md5 in exists_results:
        with open('results_v2/raw_results/level11/' + md5, 'r') as fraw:
            try:
                rsp = fraw.read()
                rsp_modified = rsp.replace(' | ','" , "')
                rsp_modified = rsp_modified.replace(' |\n', '" , "')
                # print(rsp_modified)
                # print(type(rsp_modified))
                data = json.loads(rsp_modified)
                save_data = {}
                flag_event = 0
                for key, value in data.items():
                    # print(rsp)
                    try:
                        save_data[key] = value
                        flag_event = 1
                    except:
                        save_data[key] = ""

                if flag_event == 0:
                    stat_dict['no_event'] += 1
                    filtered_records['no_event'][md5] = rsp
                    continue

                # save_data_fin = {}
                for key,value in save_data.items():
                    # event_fin = []
                    for event in value:
                        fields = event.split(';')
                        if len(fields) != 3 and '' not in fields:
                            stat_dict['invalid_format'] += 1
                            if md5 not in filtered_records['invalid_format']:
                                filtered_records['invalid_format'][md5] = []
                            filtered_records['invalid_format'][md5].append(event)
                            continue
                        s, r, o = [_.strip(' ') for _ in fields]
                        stat_dict['listed_triplets'] += 1

                        if r.lower() not in level1_name2id:
                            stat_dict['unmatched_relation'] += 1
                            if md5 not in filtered_records['unmatched_relation']:
                                filtered_records['unmatched_relation'][md5] = []
                            filtered_records['unmatched_relation'][md5].append(event)
                            continue
                        r_id = level1_name2id[r.lower()]
                        r_choice = dict_id2ont[r_id]['choice']

                        if s.lower() in UNKNOWN_ACTOR or o.lower() in UNKNOWN_ACTOR:
                            stat_dict['unknown_actor'] += 1
                            if md5 not in filtered_records['unknown_actor']:
                                filtered_records['unknown_actor'][md5] = []
                            filtered_records['unknown_actor'][md5].append(event)
                            continue
                        stat_dict['level1_events'] += 1
                        # event_fin.append(event + "; " + r_id)
                        id_list = trunk_id[md5]
                        key = key.replace("[", "")
                        key = key.replace("]", "")
                        id_trunk = int(key)
                        id_string = id_list[id_trunk]
                        key_ = id_string[id_string.find('/') + 1:]

                        event_fin = event + "; " + md5 + "; " + key_ + "; " + str(id_cnt)
                        id_cnt += 1
                        if r_id not in all_event:
                            all_event[r_id] = [event_fin]
                        else:
                            l = all_event[r_id]
                            l.append(event_fin)
                            all_event[r_id] = l
                    # save_data_fin[key] = event_fin

                with open("results_v2/clean_results/level1/events/" + md5 + '.json', 'w') as file:
                    json.dump(save_data, file, indent=2)
                with open("results_v2/clean_results/level1/all_events_.json", 'w') as file:
                    json.dump(all_event, file, indent=2)
            except:
                print(md5)
                pass

            json.dump(stat_dict, open('results_v2/clean_results/level1/' + 'stats_.json', 'w'), indent=4)
            json.dump(filtered_records, open('results_v2/clean_results/level1/' + 'filtered_.json', 'w'), indent=4)