import openai
import json
import os
import time
from tqdm import tqdm
import pandas as pd

API_KEY = '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'
openai.api_key = API_KEY

model_id = 'gpt-4'

def chatgpt_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation,
        max_tokens=512
    )
    return response

def chat(content):
    conversations = []
    # system, user, assistant
    conversations.append({'role': 'user', 'content': content})
    # print(conversations)
    conversations = chatgpt_conversation(conversations)
    return conversations.choices[0].message.content

def get_prompt2_relation(prompt, r_id):
    doc_list = json.load(open('MIDEAST_doc_v2.json', 'r'))
    dict_hier_id = json.load(open('dict_hier_id.json', 'r'))
    dict_id2ont = json.load(open('dict_id2ont_choice.json', 'r'))

    level2_ids = list(dict_hier_id[r_id].keys())
    level2_choices = [dict_id2ont[idx]['choice'] for idx in level2_ids]
    level2_description = []
    for idx in level2_ids:
        des = dict_id2ont[idx]['choice'] + " means: " + dict_id2ont[idx]['description']
        level2_description.append(des)

    rules = [
        "1. Original structured events, and corresponding news trunk are given.",
        " There are some structured events here, along with a news trunk they collectively share. In format: [corresponding news trunk] :: \n [event actor 1 (subject); event relation; event actor 2 (object)] . All events have same event relation",
        "2. All sub-relation candidates of the original event relation are also given.",
        "3. Based on the news trunk, only choose one best sub-relation for each event from the given candidate list that best matches the article, subject and object. The answer can be 'Not specified'."
    ]
    prompt_rules = f'You are an assistant to perform structured event extraction from news articles with following rules:\n' + '\n'.join(
        rules)

    prompt_example = "For example, given the structured event and corresponding news trunk: [Egypt; Express intent to cooperate; Lebanon] -- [(MENAFN- Daily News Egypt)\nEgypt is committed to enforcing economic cooperation with Lebanon, President Abdel Fattah Al-Sisi said during his meeting with Lebanese parliamentary speaker Nabih Berri.]\n" + \
                     "And given the sub-relation candidate list: Not specified; Express intent to engage in material cooperation; Express intent to engage in diplomatic cooperation; Express intent to yield or concede; Express intent to mediate.\n" + \
                     "\nChoose by rules, the sub-relation in this example is:\nExpress intent to engage in material cooperation. \n\n"

    cnt = 1
    prompt_instruction = ''
    for key in prompt:
        if key == 'null':
            break
        input_list = key.split(";")
        input_event = "; ".join(input_list[0:3])
        input_event = "[" + input_event + "]"
        input_md5 = input_list[3].strip()
        article = doc_list[input_md5]
        input_par = input_list[4].strip()
        input_par = input_par[1:-1]
        input_num = input_par.split(":")
        if input_num[0] == '0':
            input_trunk = article['Title']
        else:
            i = int(input_num[1])
            j = int(input_num[2])+1
            input_trunk = article['Text'][i:j]
            input_trunk = '[' + ''.join(input_trunk) + '] :: \n'
        if cnt == 1:
            prompt_instruction += 'given the news trunk: ' + input_trunk
        prompt_instruction += 'given the structured event' + str(cnt) + ': ' + input_event + '\n'
        cnt += 1

    cnt -= 1
    prompt_instruction = 'There are ' + str(cnt) + ' structured events.\n Now ' + prompt_instruction + "And given the sub-relation candidate list: Not specified; " + '; '.join(
                                                                                      level2_choices) + '.\n' + \
                                                                                      "And given the description for candidates: " + ".".join(level2_description) + '.\n' + \
                                                                                      "\nChoose by rules, the sub-relation in these event are:"
    # print(prompt_instruction)
    return '\n'.join([prompt_rules, prompt_example, prompt_instruction])

def get_results():
    cnt = 0
    jread = json.load(open('results_v2/clean_results/level1/all_events_duplicated.json', 'r'))
    exists_results = os.listdir('results_v2/raw_results/level2')

    for key, value in jread.items():
        loop = len(value)
        i = 0
        while loop > 0:
            input1 = input2 = input3 = input4 = input5 = input6 = input7 = input8 = input9 = input10 = input11 = input12 = input13 = input14 = input15 = input16 = input17 = input18 = input19 = input20 =  "null"
            j = 0
            while 1:
                try:
                    while value[i * 10 + 0 + j].split(";")[5] in exists_results:
                        j += 1
                    input1 = value[i * 10 + 0 + j]
                except:
                    break
                try:
                    if value[i * 10 + 1 + j].split(";")[5] in exists_results:
                        j += 1
                    input2 = value[i * 10 + 1 + j]
                except:
                    break
                try:
                    if value[i * 10 + 2 + j].split(";")[5] in exists_results:
                        j += 1
                    input3 = value[i * 10 + 2 + j]
                except:
                    break
                try:
                    if value[i * 10 + 3 + j].split(";")[5] in exists_results:
                        j += 1
                    input4 = value[i * 10 + 3 + j]
                except:
                    break
                try:
                    if value[i * 10 + 4 + j].split(";")[5] in exists_results:
                        j += 1
                    input5 = value[i * 10 + 4 + j]
                except:
                    break
                try:
                    if value[i * 10 + 5 + j].split(";")[5] in exists_results:
                        j += 1
                    input6 = value[i * 10 + 5 + j]
                except:
                    break
                try:
                    if value[i * 10 + 6 + j].split(";")[5] in exists_results:
                        j += 1
                    input7 = value[i * 10 + 6 + j]
                except:
                    break
                try:
                    if value[i * 10 + 7 + j].split(";")[5] in exists_results:
                        j += 1
                    input8 = value[i * 10 + 7 + j]
                except:
                    break
                try:
                    if value[i * 10 + 8 + j].split(";")[5] in exists_results:
                        j += 1
                    input9 = value[i * 10 + 8 + j]
                except:
                    break
                try:
                    if value[i * 10 + 9 + j].split(";")[5] in exists_results:
                        j += 1
                    input10 = value[i * 10 + 9 + j]
                except:
                    break
                try:
                    if value[i * 10 + 10 + j].split(";")[5] in exists_results:
                        j += 1
                    input11 = value[i * 10 + 10 + j]
                except:
                    break
                try:
                    if value[i * 10 + 11 + j].split(";")[5] in exists_results:
                        j += 1
                    input12 = value[i * 10 + 11 + j]
                except:
                    break
                try:
                    if value[i * 10 + 12 + j].split(";")[5] in exists_results:
                        j += 1
                    input13 = value[i * 10 + 12 + j]
                except:
                    break
                try:
                    if value[i * 10 + 13 + j].split(";")[5] in exists_results:
                        j += 1
                    input14 = value[i * 10 + 13 + j]
                except:
                    break
                try:
                    if value[i * 10 + 14 + j].split(";")[5] in exists_results:
                        j += 1
                    input15 = value[i * 10 + 14 + j]
                except:
                    break
                try:
                    input16 = value[i * 10 + 15]
                except:
                    break
                try:
                    input17 = value[i * 10 + 16]
                except:
                    break
                try:
                    input18 = value[i * 10 + 17]
                except:
                    break
                try:
                    input19 = value[i * 10 + 18]
                except:
                    break
                try:
                    input20 = value[i * 10 + 19]
                except:
                    break
                break

            loop -= 10
            if input1 == 'null':
                continue
            in_prompt = [input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15]

            try:
                msg = get_prompt2_relation(in_prompt, input1.split(";")[6].strip(' '))
                print(msg)
                #content = chat(msg)
                #print(content)
                #content_list = content.split("\n")
                '''cnt = 0
                for event_level2 in content_list:
                    level1_strip = in_prompt[cnt].split(";")
                    level1_id = level1_strip[-2]
                    event_level2 = event_level2 + '\n' + level1_id
                    with open('results_v2/raw_results/level2/' + level1_id, 'w') as fresult:
                        fresult.write(event_level2)
                        cnt += 1
                print(cnt)'''

            except Exception as e:
                level1_strip = in_prompt[cnt].split(";")
                level1_id = level1_strip[-1]
                with open('results_v2/raw_results/errors/' + level1_id, 'a') as ferror:
                    ferror.write('\n---\n')
                    ferror.write(str(e))
                    print(cnt)
                    break



    '''version = 'v1'
    level1_event_df = pd.read_csv('results_' + version + '/level1/' + 'events.csv', sep='\t', dtype='string')

    start_rowid = 0
    end_rowid = 1
    if end_rowid == 0 or end_rowid > len(level1_event_df) - 1:
        end_rowid = len(level1_event_df) - 1

    exists_results = os.listdir('raw_results_v2/results_level2')

    for idx, row in tqdm(level1_event_df.iterrows(), total=len(level1_event_df)):
        if str(idx) in exists_results:
            # print('check')
            continue

        if idx < start_rowid or idx > end_rowid:
            # print('out')
            continue
        time.sleep(1)
        try:
            md5 = row['Md5']
            article = doc_list[md5]
            msg = get_prompt2_relation(article, row)
            content = chat(msg)
            print(content)
            with open('raw_results_v2/results_level2/' + str(idx), 'w') as fresult:
                fresult.write(content)
                cnt += 1

        except Exception as e:
            with open('raw_results_v2/errors/' + str(idx), 'a') as ferror:
                ferror.write('\n---\n')
                ferror.write(str(e))
                print(cnt)
                break
    print(cnt)'''

if __name__ == '__main__':
    get_results()