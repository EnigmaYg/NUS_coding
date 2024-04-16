import os
import json
import torch
import argparse
import json
import sys
import math
import numpy
import random
from tqdm import tqdm
import pandas as pd
import re
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from time import sleep


def extract_words(value):
    words_pattern = re.compile(r'\b\w{2,}\b')
    words = words_pattern.findall(value)
    if len(words) == 0:
        return None
    else:
        result_string = ' '.join(words)
        return result_string


def parse_outputs(outputs):
    match = re.match(r'([A-Z])\b', outputs)
    pattern = r'\[([^]]+)\]'
    matches_ = re.findall(pattern, outputs)
    flag = 0
    if match:
        parse_result = [match.group(0), extract_words(outputs)]
        flag = 1
    elif matches_:
        parse_result = []
        for match_word in matches_:
            if match_word.isalpha() and match_word.isupper():
                parse_result.append(match_word)
                flag = 1
        parse_result.append(extract_words(outputs))
    else:
        parse_result = [None, outputs]
    if flag == 0:
        parse_result = [None, outputs]
    return parse_result


def generate_prompt(row, summary_list_clean, Md52timid, classification, aligned, complementary):
    def format_events_focus(events):

        sum_all = []
        for event in events:
            event = event[0]
            sum_i = []
            num_list = []
            for image_i in classification[event].keys():
                if classification[event][image_i] == "aligned":
                    num_i = aligned[event][image_i]
                    if num_i not in num_list:
                        num_list.append(num_i)
                        sum_i.append(summary_list_clean[event][num_i].strip('\n'))
            if sum_i != []:
                sum_i = "* " + "\n* ".join(sum_i)
                sum_all.append("[Date]" + str(Md52timid[event][0]) + ":\n" + sum_i)

        return "\n".join(sum_all)

    def format_events_related(events):

        sum_all = []
        for event in events:
            event = event[0]

            num_list = []
            compl_list = []
            for image_i in classification[event].keys():
                if classification[event][image_i] == "aligned":
                    print(event, end=" ")
                    print(image_i, end=" ")

                    num_i = aligned[event][image_i]
                    print(num_i)
                    if num_i not in num_list:
                        num_list.append(num_i)
                elif classification[event][image_i] == "complementary":
                    print(event, end=" ")
                    print(image_i, end=" ")
                    content_compl_i = complementary[event][image_i]
                    print(content_compl_i)
                    compl_list.append(content_compl_i)

            compl_list_str = '\n'.join(compl_list)

            sum_i = []
            for num_i_sum in range(len(summary_list_clean[event])):
                if num_i_sum == 0 or num_i_sum in num_list:
                    continue
                else:
                    sum_i.append(summary_list_clean[event][num_i_sum].strip('\n'))

            sum_i = "* " + "\n* ".join(sum_i)

            if compl_list != []:
                sum_i = sum_i + '\n' + compl_list_str

            sum_all.append("[Date]" + str(Md52timid[event][0]) + ":\n" + sum_i)

        return "\n".join(sum_all)

    rules = [
        "1. The atomic event is the basic unit describing a specific event, typically presented in the form of a quadruple (S, R, O, T), where S represents the subject, R represents the relation, O represents the object, and T represents the relative time.\n",
        "2. Please remember the meanings of the following identifiers: [Query] represents the event to be predicted in the form of (S, O, T). [Key Events] represents the list of relatively important historical events that have a significant impact on the forecast. [Related events] represents the list of supplementary event that provides additional information and relevant context. [Options] represents the candidate set of the missing object.\n"
        "3. When formulating the ultimate prediction, the preeminent factor to be meticulously weighed and scrutinized is the [Key Events]. Complementing this paramount consideration is the [Related events], which, though ancillary in nature, serves as a valuable adjunct, furnishing pertinent contextual details and auxiliary insights to fortify the predictive analysis.\n",
        "4. Given a query of (S, R, T) in the future and the list of historical events until t, event forecasting aims to predict the missing object.",
        # "The inputs are as follows:\n"
    ]
    prompt_rules = "You are an assistant to perform event forecasting with the following rules:\n" + ''.join(rules)

    subject = row['Subject']
    relation = row['Relation']
    correct_object = row['Object']
    time = row['timid']

    ###
    # nearest_events = format_events(eval(row['Nearest_events_summary']))
    # further_events = format_events(eval(row['Further_events_summary']))
    # related_facts = format_events(eval(row['Related_facts_summary']))

    related_facts_list = eval(row['Nearest_events_summary']) + eval(row['Further_events_summary']) + eval(
        row['Related_facts_summary'])
    print(related_facts_list)
    related_facts_dic = {}
    for re_event in related_facts_list:
        related_facts_dic[re_event] = Md52timid[re_event][0]

    related_facts_list_sort = sorted(related_facts_dic.items(), key=lambda kv: (kv[1], kv[0]))
    print(related_facts_list_sort)

    print('-------------')
    focus_facts = format_events_focus(related_facts_list_sort)
    related_facts = format_events_related(related_facts_list_sort)
    print('-------------')

    ###
    candidates = eval(row['Candidates'])
    group_num = row['groupid']
    # id = row['ID']

    options = candidates + [correct_object]
    random.shuffle(options)

    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapped_options = {option: option_letters[i] for i, option in enumerate(options)}

    prompt = (
        "[Query]: ({}, {}, {})\n"
        "[Key Events]: \n{}\n"
        "[Related Events]: \n{}\n"
        "[Options]:\n{}\n"
        "Please provide a correct option in the following format:\n The correct option is: [the letter of the correct option without any explanation]."
    ).format(subject, relation, time, focus_facts, related_facts,
             "\n".join([f"{letter}: {option}" for option, letter in mapped_options.items()]))



    return prompt, prompt_rules, [mapped_options[correct_object], correct_object], group_num


def prefix_allowed_tokens_fn(batch_id, input_ids):
    allowed_tokens = list(range(65, 91))
    return allowed_tokens
    # if len(input_ids) > 0:
    #     if input_ids[-1] in allowed_tokens:
    #         return allowed_tokens
    # return []


# @torch.inference_mode()
def main():
    '''print(f"args.max_tokens:{args.max_tokens}")
    print(f"args.temperature:{args.temperature}")'''
    # sys.stdout = Logger(stream = sys.stdout)
    # sys.stderr = Logger(stream = sys.stderr)
    df = pd.read_csv('/mnt/hdd/hxli/MM_Event_forecasting/Code_For_EF/gemini/final_test.csv', sep="\t", dtype={"Relation_id": str})
    df = df.loc[df['timid'] > 365 * 6]
    with open('/mnt/hdd/hxli/MM_Event_forecasting/Code_For_EF/Data_gemini/sum_gemini_all.json', 'r') as file:
        summary_list_clean = json.load(file)
    with open('/mnt/hdd/hxli/MM_Event_forecasting/Code_For_EF/gemini/Md52timid.json', 'r') as f:
        Md52timid = json.load(f)

    with open('/mnt/hdd/hxli/MM_Event_forecasting/Code_For_EF/Data_gemini/classification_gemini_all.json', 'r') as f:
        classification = json.load(f)
    with open('/mnt/hdd/hxli/MM_Event_forecasting/Code_For_EF/Data_gemini/aligned_gemini_all.json', 'r') as f:
        aligned = json.load(f)
    with open('/mnt/hdd/hxli/MM_Event_forecasting/Code_For_EF/Data_gemini/complementary_gemini_all_new.json', 'r') as f:
        complementary = json.load(f)

    total_num = 0
    true_num = 0
    true_num_group = [0] * len(df['groupid'].unique().tolist())
    result = []

    wrong_event = []

    with tqdm(total=df.shape[0], desc="Event Forecasting") as pbar:
        for index, row in df.iterrows():
            print()
            print("-------------------------------")
            id = row['ID']
            prompt, prompt_rules, correct_option, group_num = generate_prompt(row, summary_list_clean, Md52timid,
                                                                              classification, aligned, complementary)
            model_dir = "/mnt/hdd/llm/llama2_hf/llama-2-7b"
            # pipe = pipeline("conversational", model_dir)

            model = AutoModelForCausalLM.from_pretrained(model_dir).cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_dir,legacy=False,use_fast=True)
            prompt = prompt + "\n" + prompt_rules
            messages = [[{"role": "system", "content": prompt_rules,}, {"role": "user", "content": prompt},]]
            print("**")
            print(prompt)
            print("**")

            # tokenized_chat = tokenizer.default_chat_template(messages, tokenize=True,
            #                                                 return_tensors="pt")

            inputs = tokenizer(prompt,truncation=True,max_length=512, return_tensors="pt").to('cuda')
            outputs = model.generate(**inputs, max_new_tokens=128, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

            try:
                # response = pipe(messages)
                response = tokenizer.decode(outputs[0])
            except:
                wrong_event.append(index)
                continue

            '''conv = get_default_conv_template(args.model_path).copy()
            conv.append_message(conv.system, prompt_rules)
            print("**")
            print(prompt)
            print("**")
            print(prompt_rules)
            print("**")
            print(correct_option)
            print("**")
            print(group_num)
            print("**")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()'''
            '''input_ids = tokenizer([prompt]).input_ids
                            output_ids = model.generate(
                                torch.as_tensor(input_ids).cuda(),
                                do_sample=True,
                                temperature=0.001,
                                max_new_tokens=args.max_tokens,
                            )
                            if model.config.is_encoder_decoder:
                                output_ids = output_ids[0]
                            else:
                                output_ids = output_ids[0][len(input_ids[0]):]
                            response = tokenizer.decode(output_ids, skip_special_tokens=True,
                                                        spaces_between_special_tokens=False)'''



            print(correct_option)
            print("-------------------------------")
            print()

            if response == None:
                pbar.update(1)
                wrong_event.append(index)
                print()
                print(row)
                print()
                continue

            # print(f"prompt_rules:\n{prompt_rules}")
            # print(f"prompt:\n{prompt}")
            # print(f"response:\n{response}")
            total_num += 1
            pbar.update(1)
            '''print("**")
            print(response)
            print("**")'''
            response = parse_outputs(response)
            if any(item in correct_option for item in response):
                true_num += 1
                true_num_group[group_num] += 1
            print("**")
            print(correct_option)
            print("**")
            print(response)
            print("**")
            print(true_num)

            result.append(
                {
                    "answer": correct_option,
                    "predict": response,
                    "group_num": group_num,
                    "ID": id
                }
            )

            if total_num % 200 == 0:
                print(f"current acc:{true_num / total_num}")

        group_dict = df['groupid'].value_counts().to_dict()
        print(f"total_num:{total_num} == {sum(list(group_dict.values()))}")

        precision = true_num / total_num
        print(f"precision:{precision}")
        for i in range(len(true_num_group)):
            precision_group = true_num_group[i] / group_dict[i]
            print(f"precision{i}:{precision_group}")

        with open("result_rule_text_llama.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print('save in ./result_rule_text_w_mm.json')

        wrong_ev = {}
        wrong_ev['wrong_ev_index'] = wrong_event
        with open("text_llama_wrong.json", "w") as f:
            json.dump(wrong_ev, f)


if __name__ == "__main__":
    '''parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--input_path', type=str,
                        default='/mnt/hdd/hxli/MM_Event_forecasting/Code_For_EF/gemini/final_test.csv',
                        help='input data directory')
    parser.add_argument('--output_path', type=str,
                        default='/home/zmyang/zmyang_workplace/MM2024/open_source/graph_rule_based/vicuna/',
                        help='output data directory')
    # add_model_args(parser)
    # parser.add_argument("--data_path", type=str, default="/mnt/hdd/hxli/Datasets/Dataset_changhe/MIDEAST-TE-mini/test.csv")
    # parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    # parser.add_argument("--repetition_penalty", type=float, default=0.4)
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()'''
    # with torch.device.device(2):
    main()
