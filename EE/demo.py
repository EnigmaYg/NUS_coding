import pandas as pd
from tqdm import tqdm
import json
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import os

def calculate_tokens():
    doc_list = json.load(open('MIDEAST_doc.json', 'r'))
    print(len(doc_list))
    # print(len(str(doc_list['eb93470bb4fad9b12222ef00d48e0269']['Text']).split(' ')))
    cnt_del = cnt_all = 0
    del_tokens = 0
    save_tokens = 0
    whole_tokens = 0

    for key, value in doc_list.items():
        cnt = cnt_del
        raw_tokens = 0
        total_tokens = 0


        raw_tokens_title = value['Title'].split(' ')
        raw_tokens_title_length = len(raw_tokens_title)
        raw_tokens += raw_tokens_title_length

        total_tokens_title_length = raw_tokens_title_length
        total_tokens += total_tokens_title_length
        total_tokens_text = str(value['Text']).split(' ')
        total_tokens_text_length = len(total_tokens_text)
        total_tokens += total_tokens_text_length

        whole_tokens += total_tokens
        for i in range(1, len(value['Text']) + 1):
            raw_tokens_text = str(value['Text'][:i]).split(' ')
            raw_tokens_text_length = len(raw_tokens_text)
            raw_tokens += raw_tokens_text_length
            if raw_tokens > 1024:
                i -= 1
                cnt_del += 1
                break
            if i != len(value['Text']):
                raw_tokens -= raw_tokens_text_length

        del_tokens += total_tokens - raw_tokens
        save_tokens += raw_tokens

        if cnt == cnt_del:
            cnt_all += 1

    print("Number of articles that are more than 1024 tokens : {}".format(cnt_del))
    print("Number of articles that are less than 1024 tokens : {}".format(cnt_all))
    print("Number of tokens for all : {}".format(whole_tokens))
    print("Number of tokens have saved : {}".format(save_tokens))
    print("Number of tokens have deleted : {}".format(del_tokens))

def calculate_relation():
    level1_event = pd.read_csv('/Users/milesyzm/Downloads/results_v1/level1/events.csv', sep='\t', dtype='string')
    level1_relation = {}
    level1_md5 = {}
    for idx, row in tqdm(level1_event.iterrows(), total=len(level1_event)):
        s, r_id, r_choice, o, md5 = row
        if r_choice not in level1_relation:
            level1_relation[r_choice] = 1
        else:
            level1_relation[r_choice] += 1
        if md5 not in level1_md5:
            level1_md5[md5] = 1
        else:
            level1_md5[md5] += 1
    sorted_level1_relation = dict(sorted(level1_relation.items(), key=lambda item: item[1]))
    sorted_level1_md5 = dict(sorted(level1_md5.items(),  key=lambda item: item[1], reverse=True))
    print(sorted_level1_relation)
    print(sorted_level1_md5)


def json2csv():
    dic = {}
    all_events = []
    events_json = json.load(open('results_v2/clean_results/level1/all_events_id.json', 'r'))

    for key, value in events_json.items():
        for content in value:
            try:
                content_list = content.split('; ')
                subject = content_list[0]
                relation_choice = content_list[1]
                relation_id = key
                object = content_list[2]
                Md5 = content_list[3]
                trunk = content_list[4]
                id = content_list[5]
                all_events.append([subject, relation_id, relation_choice, object, Md5, trunk, id])
            except:
                print(content)
    all_events_df = pd.DataFrame(all_events,
                                 columns=['Subject', 'Relation_id', 'Relation_choice', 'Object', 'Md5', 'Trunk', 'ID'],
                                 dtype='string')
    all_events_df.to_csv(path_or_buf='results_v2/clean_results/level1/all_events.csv',
                         sep='\t', index=False)


def csv2json():
    dic = {}
    csvfile = pd.read_csv('results_v2/clean_results/all_events_fin.csv', sep='\t',
                                  dtype='string')
    for idx, row in tqdm(csvfile.iterrows(), total=len(csvfile)):
        s, r_id, r_choice, o, md5, trunk, id = row
        content = s + "; " + r_choice + "; " + o + "; " + md5 + "; " + trunk + "; " + id
        try:
            content.replace("'", "\"")
            r_id.replace("'", "\"")
        except:
            print(row)
            continue
        if r_id not in dic:
            dic[r_id] = [content]
        else:
            dic[r_id].append(content)
    with open("split/all_events_id.json", 'w') as file:
        json.dump(dic, file, indent=2)

def modify_id():
    dic = {}
    all_events = []
    events_json = json.load(open('results_v2/clean_results/level1/all_events.json', 'r'))
    new_id = 0
    for key, value in events_json.items():
        l = []
        for content in value:
            try:
                content_list = content.split('; ')
                subject = content_list[0]
                relation_choice = content_list[1]
                relation_id = key
                object = content_list[2]
                Md5 = content_list[3]
                trunk = content_list[4]
                id = str(new_id)
                new_id += 1
                ll = [subject, relation_choice, object, Md5, trunk, id]
                l.append( '; '.join(ll) )
            except:
                print(content)
                break
        dic[key] = l
    with open("results_v2/clean_results/level1/all_events_id_.json", 'w') as file:
        json.dump(dic, file, indent=2)


def get_duplicated_trunk():
    events_json = json.load(open('results_v2/clean_results/level1/all_events_id.json', 'r'))
    dic = {}
    md5 = ''
    trunk = ''
    flag = 0
    id = 0
    cnt = 0
    for key, value in events_json.items():
        l = []
        for content in value:
            content_list = content.split('; ')
            if md5 == content_list[3] and trunk == content_list[4]:
                flag = 1
            else:
                flag = 0
                if len(l) >= 6:
                    dic[id] = l
                    cnt += len(l)
                    id += 1
            if flag == 0:
                content = content + "; " + key
                l = [content]
            else:
                content = content + "; " + key
                l.append(content)
            md5 = content_list[3]
            trunk = content_list[4]
    with open("results_v2/clean_results/level1/all_events_duplicated.json", 'w') as file:
        json.dump(dic, file, indent=2)
    print(cnt)


def merge_levels():
    level1_df = pd.read_csv('results_v2/clean_results/level1/all_events.csv', sep='\t', dtype='string')
    level2_df = pd.read_csv('results_v2/clean_results/level2/all_events.csv', sep='\t', dtype='string')
    level3_df = pd.read_csv('results_v2/clean_results/level3/all_events.csv', sep='\t', dtype='string')
    unique_event_in_level1 = level1_df[~level1_df['ID'].isin(level2_df['ID']) ]
    unique_event_in_level2 = level2_df[~level2_df['ID'].isin(level3_df['ID'])]
    all_events = pd.concat([unique_event_in_level1, unique_event_in_level2, level3_df], ignore_index=True)
    all_events = all_events.sort_values(by=['ID'], key=lambda x: x.astype(int), ignore_index=True)
    all_events.to_csv(path_or_buf='results_v2/clean_results/all_events_fin.csv', sep='\t', index=False)


def sort_key(element):
    return int(element.split('; ')[5])


def merge_operations():
    with open('results_v2/clean_results/level1/all_events__.json', 'r') as file:
        events_json_new = json.load(file)
    with open('results_v2/clean_results/level1/all_events_id_.json', 'r') as file:
        events_json_old = json.load(file)
    cnt = 0
    count = 8731
    c = 0
    list_old = []
    for key, value in events_json_old.items():
        for content in value:
            list_old.append(content)

    sorted_list_old = sorted(list_old, key=sort_key)
    # print(sorted_list_old)
    list_old_no_id = []
    for i in range(len(sorted_list_old)):
        content = '; '.join(sorted_list_old[i].split('; ')[0:5])
        list_old_no_id.append(content)

    dic = {}
    for key, value in events_json_new.items():
        l = []
        for content in value:
            if content in list_old_no_id:
                c += 1
                ID = str(list_old_no_id.index(content))
                content += '; '
                content += ID
                l.append(content)
            else:
                cnt += 1
                content += '; '
                content += str(count)
                count += 1
                l.append(content)
        dic[key] = l

    with open("results_v2/clean_results/level1/all_events_id.json", 'w') as file:
        json.dump(dic, file, indent=2)


def Split():
    cnt = 22876
    with open('split/r_id_with_md52.json', 'r') as file:
        events_json = json.load(file)
    with open('split/r_id2.json', 'r') as file:
        id_json = json.load(file)
    with open('split/md5_with_id2.json', 'r') as file:
        md5_json = json.load(file)
    # use dp to collect all event types

    l_key = list(id_json.keys())

    l_id = [str(x) for x in range(0, 39657)]
    # print(len(l_id))

    results = []

    count_ = 0
    for key, value in events_json.items():
        if not l_key:
            break
        if key in l_key:
            key_md5, value_id = next(iter(value.items()))
            count_ += len(value_id)
            for tid in value_id:
                l_id.remove(tid)
            l_key.remove(key)
            for second_key, second_value in events_json.items():
                if key_md5 in second_value:                 # 当前的article 在别的r_id中也有事件类型
                    if second_key in l_key:                 # 且该类型暂未被收录
                        l_key.remove(second_key)
                try:
                    second_id = events_json[second_key][key_md5]
                    for tid_ in second_id:
                        l_id.remove(tid_)
                    count_ += len(second_id)
                except:
                    pass
            results.append(key_md5)
        else:
            continue

    # print(len(l_id))

    # print(l_key)
    # print(len(results))

    count = 0
    for key, value in md5_json.items():
        if key in results:
            count += len(value)

    save = []
    for key, value in id_json.items():
        if value < 4:
            for md5 in events_json[key]:
                if md5 not in results:
                    save.append(md5)
    # print(len(save))
    for key, value in md5_json.items():
        if key not in results and key not in save:
            results.append(key)
            count += len(value)
        if count >= cnt:
            break
    print(f"Amount of Triplet in Train set {count}")

    results2 = []
    cnt2 = 2859
    count = 0
    for key, value in md5_json.items():
        if key not in results:
            if '[' in key:
                continue
            results2.append(key)
            count += len(value)
        if count >= cnt2:
            break
    print(f"Amount of Triplet in Eval set {count}")

    count = 0
    results3 = []
    for key, value in md5_json.items():
        if key not in results and key not in results2:
            if '[' in key:
                continue
            results3.append(key)
            count += len(value)
    print(f"Amount of Triplet in Test set {count}")
    dic_save = {}
    dic_save["Train"] = results
    dic_save["Test"] = results2
    dic_save["Evaluation"] = results3
    with open("split/split2.json", 'w') as file:
         json.dump(dic_save, file, indent=2)
    print(f"Amount of Article in Train set {len(results)}")
    print(f"Amount of Article in Eval set {len(results2)}")
    print(f"Amount of Article in Test set {len(results3)}")
    s = set(list(md5_json.keys()))
    s1 = set(results)
    s1.update(results2)
    s1.update(results3)
    # print(s.difference(s1))
    # print(len(md5_json))
    # print(count_)


def object_distribution():
    dic = {}
    csvfile = pd.read_csv('results_v2/clean_results/all_events_fin.csv', sep='\t',
                          dtype='string')
    for idx, row in tqdm(csvfile.iterrows(), total=len(csvfile)):
        s, r_id, r_choice, o, md5, trunk, id = row
        content = s + "; " + r_choice + "; " + o + "; " + md5 + "; " + trunk + "; " + id
        try:
            content.replace("'", "\"")
            r_id.replace("'", "\"")
        except:
            print(row)
            continue
        if len(md5) != 32:
            continue
        if md5 not in dic:
            dic[md5] = [s]
        elif s not in dic[md5]:
            dic[md5].append(s)
        if o not in dic[md5]:
            dic[md5].append(o)

    cnt_in = 0
    cnt_notin = 0
    cnt_subin = 0
    dic_sub = {}
    with open('MIDEAST_doc_v2.json', 'r') as file:
        events_json = json.load(file)
    for key, value in dic.items():
        for v in value:
            if v in events_json[key]["Title"]:
                dic_sub[v] = "0"
                cnt_in += 1
                continue
            if v in ' '.join(events_json[key]["Text"]):
                dic_sub[v] = "0"
                cnt_in += 1
            else:
                if v.lower() in ' '.join(events_json[key]["Title"]).lower():
                    dic_sub[v] = "0"
                    cnt_in += 1
                    continue
                elif v.lower() in ' '.join(events_json[key]["Text"]).lower():
                    dic_sub[v] = "0"
                    cnt_in += 1
                    continue
                result_ = find_substring(v, ' '.join(events_json[key]["Text"]).lower())
                if result_ != ' ':

                    cnt_subin += 1
                    continue
                cnt_notin += 1

    print(f"entities that in the doc are {cnt_in}")
    print(f"substring of entities that in the doc are {cnt_subin}")
    print(f"entity and its substring neither is in the doc are {cnt_notin}")
    with open("split/dic_sub.json", 'w') as file:
         json.dump(dic_sub, file, indent=2)


def find_substring(c1, c2):
    c1 = c1.lower()
    c2 = c2.lower()
    c1_words = pos_tags_(c1)
    # 将 c1 中的字符串以空格分割成单词列表
    # c1_words = c1.split()

    # 遍历 c1 的所有子串
    for i in range(len(c1_words)):
        for j in range(i + 1, len(c1_words) + 1):
            # 构建子串
            substring = ' '.join(c1_words[i:j])
            sub = substring.split()
            if len(sub) < 2:
                continue

            # 判断子串是否在 c2 中
            if substring in c2:
                return substring
    return ' '


def pos_tags_(text):
    words_before = word_tokenize(text.lower())

    # 删除停用词和标点符号
    stop_words = set(stopwords.words('english') + list(punctuation))

    filtered_words = [w for w in words_before if w not in stop_words]

    # 词性标注，保留名词
    pos_tags = nltk.pos_tag(filtered_words)
    filtered_pos_tags = [(word, pos) for word, pos in pos_tags if pos.startswith('NN')]

    # 打印只包含名词的列表
    result_words = [word for word, pos in filtered_pos_tags]
    return result_words

    # 词性标注

    # pos_tags_ = nltk.pos_tag(words)
    # print(pos_tags_)


if __name__ == '__main__':
    # calculate_tokens()
    # calculate_relation()
    # csv2json()
    # json2csv()
    # modify_id()
    # get_duplicated_trunk()
    # merge_levels()
    # merge_operations()
    # Split()
    # object_distribution()

    with open("Eval/md5_list.json", 'r') as f:
        md5_ = json.load(f)
    with open("dict_id2ont_choice.json", 'r') as f:
        choice = json.load(f)

    md5_list = md5_["md5"]

    dic = {}
    csvfile = pd.read_csv('Eval/EGIRIS_ce_final.csv', sep='\t',
                          dtype='string')
    cnt = 0
    for idx, row in tqdm(csvfile.iterrows(), total=len(csvfile)):
        # s, o, r_id, x1, x2, x3, day, timeid, year, md5, ce_id,  md5_l = row
        s, r_id, r_choice, o, md5, day, ceid, timid, md5_l = row
        '''if r_id in choice:
            relation = choice[r_id]["choice"]
        else:
            r_id = "0" + r_id
            relation = choice[r_id]["choice"]'''
        content = s + "; " + r_id + "; " + r_choice + "; " + o + "; " + md5 + "; " + str(cnt)
        if md5 in md5_list:
            cnt += 1
        try:
            content.replace("'", "\"")
            r_id.replace("'", "\"")
        except:
            print(row)
            continue
        if md5 not in dic:
            dic[md5] = [content]
        else:
            dic[md5].append(content)
    save = {}
    for v in md5_["md5"]:
        save[v] = dic[v]
    print(cnt)

    with open("Eval/all_events_EGIRIS.json", 'w') as f:
         json.dump(save, f, indent=4)


    '''exists_results = os.listdir('results_v2/raw_results/level3')
    print(len(exists_results))
    for i in range(0,8730):
        if str(i) in exists_results:
            print(i)'''

    '''in_prompt = ["1;2;3;4;5;6"]
    with open('new/results/1', 'r') as fraw:
        content = fraw.read()
    content_list = content.split("\n")
    print(content_list)
    cnt = 0
    for evnent_level2 in content_list:
        level1_strip = in_prompt[cnt].strip(";")
        level1_id = level1_strip[-1]'''

    '''dic = {}
    csvfile = pd.read_csv('split/all_events_entity_match.csv', sep='\t',
                          dtype='string')
    for idx, row in tqdm(csvfile.iterrows(), total=len(csvfile)):
        s, r_id, r_choice, o, md5, trunk, id = row
        content = s + "; " + r_choice + "; " + o + "; " + md5 + "; " + trunk + "; " + id
        try:
            content.replace("'", "\"")
            r_id.replace("'", "\"")
        except:
            print(row)
            continue

        if md5 not in dic:
            dic[md5] = [id]
        else:
            dic[md5].append(id)
    with open("split/md5_with_id2.json", 'w') as file:
        json.dump(dic, file, indent=2)'''

    '''dic = {}
    csvfile = pd.read_csv('split/all_events_entity_match.csv', sep='\t',
                          dtype='string')
    for idx, row in tqdm(csvfile.iterrows(), total=len(csvfile)):
        s, r_id, r_choice, o, md5, trunk, id = row
        content = s + "; " + r_choice + "; " + o + "; " + md5 + "; " + trunk + "; " + id
        try:
            content.replace("'", "\"")
            r_id.replace("'", "\"")
        except:
            print(row)
            continue


        if r_id not in dic:
            dic_temp = {}
            dic_temp[md5] = [id]
            dic[r_id] = dic_temp
        else:
            if md5 not in dic[r_id]:
                dic_temp = {}
                dic_temp[md5] = [id]
                dic[r_id].update(dic_temp)
            else:
                l = dic[r_id][md5]
                l.append(id)
                dic[r_id][md5] = l
    with open("split/r_id_with_md52.json", 'w') as file:
        json.dump(dic, file, indent=2)'''
