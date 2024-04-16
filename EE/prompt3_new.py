import time

import openai
import json
import os
from tqdm import tqdm
import pandas as pd

API_KEY = ''
openai.api_key = API_KEY

model_id = 'gpt-4'

def chatgpt_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation,
        max_tokens=500
    )
    return response

def chat(content):
    conversations = []
    # system, user, assistant
    conversations.append({'role': 'user', 'content': content})
    # print(conversations)
    conversations = chatgpt_conversation(conversations)
    return conversations.choices[0].message.content


def get_prompt3_relation(article, row, dict_hier_id):
    dict_id2ont = json.load(open('dict_id2ont_choice.json', 'r'))
    level1_rowid, s, r_id, r_choice, o, md5 = row
    level3_ids = dict_hier_id[r_id]
    level3_choices = [dict_id2ont[idx]['choice'] for idx in level3_ids]
    level3_description = []
    for idx in level3_ids:
        des = dict_id2ont[idx]['choice'] + " means: " + dict_id2ont[idx]['description']
        level3_description.append(des)

    rules = [
        "1. A original structured event is given in format: event actor 1 (subject); event relation; event actor 2 (object).",
        "2. All sub-relation candidates of the original event relation are also given.",
        "3. A news article where the original structured event is extracted from is also given.",
        "4. Based on the article, only choose one best sub-relation from the given candidate list that best matches the article, subject and object. The answer can be 'Not specified'."
    ]
    prompt_rules = f'You are an assistant to perform structured event extraction from news articles with following rules:\n' + '\n'.join(
        rules)

    prompt_example = "For example, given the structured event: Egypt; Express intent to material cooperate; Lebanon.\n" + \
                     "And given the sub-relation candidate list: Not specified; Express intent to cooperate economically; Express intent to cooperate militarily; Express intent to cooperate on judicial matters; Express intent to cooperate on intelligence or information sharing.\n" + \
                     "And given the news article:\n" + \
                     "(MENAFN- Daily News Egypt)\nEgypt is committed to enforcing economic cooperation with Lebanon, President Abdel Fattah Al-Sisi said during his meeting with Lebanese parliamentary speaker Nabih Berri.\n" + \
                     "\nChoose by rules, the sub-relation in this example is:\nExpress intent to cooperate economically.\n"

    text_article = article['Title'] + '\n' + (
        '\n'.join(article['Text'][:3]) if len(article['Text']) >= 3 else '\n'.join(article['Text']))
    raw_tokens = text_article.split(' ')
    if len(raw_tokens) > 512:
        text_article = ' '.join(raw_tokens[:512])

    prompt_instruction = 'Now, given the structured event: ' + '; '.join([s, r_choice, o]) + '\n' + \
                         "And given the sub-relation candidate list: Not specified; " + '; '.join(
        level3_choices) + '.\n' + \
                         "And given the description for candidates: " + ".".join(level3_description) + '.\n' + \
                         "And given the news article:\n" + text_article + '\n' + \
                         "\nChoose by rules, the sub-relation in this example is:"

    return '\n'.join([prompt_rules, prompt_example, prompt_instruction])


if __name__ == '__main__':
    mread = json.load(open('md5list.json', 'r'))
    doc_list = json.load(open('test_dataset.json', 'r'))
    md5_list = mread['md5']

    version = 'v1'
    level2_event_df = pd.read_csv('results_' + version + '/level2/' + 'events.csv',  sep='\t', dtype='string')

    start_rowid = 0
    end_rowid = 0
    if end_rowid == 0 or end_rowid > len(level2_event_df) - 1:
        end_rowid = len(level2_event_df) - 1

    exists_results = os.listdir('raw_results_v2/results_level3')

    num_no_level3 = 0

    for idx, row in tqdm(level2_event_df.iterrows(), total=len(level2_event_df)):
        if str(idx) in exists_results:
            print('check')
            continue

        if idx < start_rowid or idx > end_rowid:
            print('out')
            continue

        dict_hier_id = json.load(open('dict_hier_id_level23.json', 'r'))
        level2_id = row['Relation_id']
        # print(level2_id)
        if level2_id not in dict_hier_id:
            num_no_level3 += 1
            continue

        try:
            time.sleep(1)
            md5 = row['Md5']
            article = doc_list[md5]
            # print(row)
            msg = get_prompt3_relation(article, row, dict_hier_id)
            content = chat(msg)
            with open('raw_results_v2/results_level3/' + str(idx), 'w') as fresult:
                fresult.write(content)

        except Exception as e:
            with open('raw_results_v2/errors/' + str(idx), 'a') as ferror:
                ferror.write('\n---\n')
                ferror.write(str(e))

print("no level 3 events: " + str(num_no_level3))
