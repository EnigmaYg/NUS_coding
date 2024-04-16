import os
import json
import torch
import argparse
import json
import sys
import os
import math
import numpy
import random
from tqdm import tqdm
import pandas as pd
import re
from openai import OpenAI
from rank_bm25 import BM25Okapi
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import os

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

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
    if match:
        parse_result = [match.group(0), extract_words(outputs)]
    else:
        parse_result = ['No', outputs]
    return parse_result


def read_dictionary(filename):
    d = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d

date2id = read_dictionary("./data/date2id.txt")
id2date = {v:k for k,v in date2id.items()}

def absolute2relative(events):
    events_relative = []
    if len(events)>0:
        for event in events:
            timid = id2date[event[3]]
            event = list(event)
            event[3] = timid
            events_relative.append(event)
    return events_relative

def generate_prompt_rules(task):
    if task == 'reasoning':
        rules = [
        "1. The atomic event is the basic unit describing a specific event, typically presented in the form of a quadruple (S, R, O, T), where S represents the subject, R represents the relation, O represents the object, and T represents the relative time.\n",
        "2. Complex Event, which is composed of a set of atomic events, describes the temporal evolution process of multiple atomic events.\n",
        "3. Please remember the meanings of the following identifiers:[Query] represents the event to be predicted in the form of (S, R, T). [Relevant Event] represents a list of atomic events relevant to the query. [Relevant News Text] represents background information about subject. [Supplementary Text] represents additional background information about subject. [Options] represents the candidate set of the missing object.\n",
        "4. The format of [Relevant News Text] is as follows: [Date]relative time: news text.\n"
        "5. You need to predict the missing object based on the [Relevant Event], [Relevant News Text] and [Query]."
        ]
        prompt_rules = "You are an assistant to perform event forecasting with the following rules:\n" + ''.join(rules)
    elif task == 'text_align':
        rules = [
        # "1. The atomic event is the basic unit describing a specific event, typically presented in the form of a quadruple (S, R, O, T), where S represents the subject, R represents the relation, O represents the object, and T represents the relative time.\n",
        # "2. Please remember the meanings of the following identifiers: [Subject] represents the subject in the query event. [Relation] represents the relation in the query event. [Event and News Summary] represents a list of news summaries that may be relevant to the subject and relation in the query event.",
        "Please remember the meanings of the following identifiers: [News Text] represents news text relevant to the subject in the current event.\n",
        ]
        prompt_rules = "You are an assistant to summarize news text with the following rules:\n" + ''.join(rules)
    elif task == 'prune_entity':
        rules = [
        "1. [Subject] represents the event subject in a specific event. [Candidate Set] represents a list of entities.\n",
        "2. You need to select the entities that may be relevant to [Subject]."
        # "3. Your response should be provided in the form of a Python list. If none is selected, ouput None. Otherwise, you must output in the following format: [entity0, entity1, entit2]."
        ]
        prompt_rules = "You are an assistant to find relevant entities with the following rules:\n" + ''.join(rules)
    elif task == 'reasoning_t':
        rules = [
        "1. The atomic event is the basic unit describing a specific event, typically presented in the form of a quadruple (S, R, O, T), where S represents the subject, R represents the relation, O represents the object, and T represents the relative time.\n",
        # "2. Complex Event, which is composed of a set of atomic events, describes the temporal evolution process of multiple atomic events.\n",
        "2. Please remember the meanings of the following identifiers: [Query] represents the event to be predicted in the form of (S, R, T). [Relevant News Text] represents background information about subject. [Options] represents the candidate set of the missing object.\n",
        "3. The format of [Relevant News Text] is as follows: [Date]relative time: news text.\n"
        "4. You need to predict the missing object based on the [Relevant News Text] and [Query]."
        ]
        prompt_rules = "You are an assistant to perform event forecasting with the following rules:\n" + ''.join(rules)

    return prompt_rules

def generate_reasoning_input_t(row, complement_summary, last = False):
    # complement_summary = sorted(complement_summary, key=lambda item: item[0],reverse=True)
    subject = row['Subject']
    relation = row['Relation']
    time = row['timid']
    correct_object = row['Object']
    candidates = eval(row['Candidates'])
    options = candidates + [correct_object]
    random.shuffle(options)
    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapped_options = [[option_letters[i],option]  for i, option in enumerate(options)]
    if last:
        input = (
                "[Query]: ({}, {}, {})\n"
                # "[Subject]: {}\n"
                "[Relevant News Text]:\n{}\n"
                "[Options]:\n{}\n"
                "You must only output the letter of the missing object without any explanation."
            ).format(subject, relation, time,  "\n".join(["[Date]" +str(text[0]) + ":\n"+text[1] for text in complement_summary]),
                    "\n".join([f"{option[0]}: {option[1]}" for option in mapped_options]))
    for option in mapped_options:
        if option[1] == correct_object:
            correct_object_letter = option[0]
            break
    return input,[correct_object_letter, correct_object],mapped_options


def generate_reasoning_input(row, search_result, complement_summary, last = False):
    def format_events(events):
        return "; ".join(["(" + ", ".join(map(str, event)) + ")" for event in events])
    
    if type(search_result)==list:
        search_result = format_events(search_result)

    subject = row['Subject']
    relation = row['Relation']
    time = row['timid']
    correct_object = row['Object']
    candidates = eval(row['Candidates'])
    options = candidates + [correct_object]
    random.shuffle(options)
    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapped_options = [[option_letters[i],option]  for i, option in enumerate(options)]
    if last:
        input = (
                "[Query]: ({}, {}, {})\n"
                # "[Subject]: {}\n"
                "[Relevant Event]:{}\n"
                "[Relevant News Text]: \n{}\n"
                "[Options]:\n{}\n"
                "You must only output the letter of the missing object without any explanation."
            ).format(subject, relation, time, search_result, "\n".join(["[Date]" +str(text[0]) + ":\n"+text[1] for text in complement_summary]),
                    "\n".join([f"{option[0]}: {option[1]}" for option in mapped_options]))
    for option in mapped_options:
        if option[1] == correct_object:
            correct_object_letter = option[0]
            break
    return input,[correct_object_letter, correct_object],mapped_options


def generate_prune_input(subject, prune_set):
    prune_list = list(prune_set)
    if subject in prune_list:
        prune_list.remove(subject)
    input = (
        "[Subject]: {}\n"
        "[Candidate Set]: {}\n"
        "Your response should be provided in the form of JSON. The example of output format is as following: [selected_entities: [entity0, entity1,...,entityN-1,entityN]]. If none is selected, ouput None."
    ).format(subject,prune_list)
    return input

def generate_prompt_text(subject,text):
    input = (
    # "[Subject]: {}\n"
    "[News Text]:{}\n"
    "You need to generate a concise summary based on the [News Text]. Your response should only include the generated summary."
    ).format(text)
    return input


def generate_prompt_graph_summary(graph_align, graph_memory):
    if len(graph_memory) != 0:
        input = (
            "[Relevant Event]:{}\n"
            "[Supplementary Text]: {}\n"
            "[Event and News Summary]: {}\n"
            "You need to summarize all the news summaries and events in [Event and News Summary]. Your response should only include the generated summary."
        ).format(','.join([str(event[0]) for event in graph_memory['graph']]),graph_memory['summary'],",".join(["\n["+ str(event[0]) + ':' + str(event[1])+']' for event in graph_align]))

    else:
        input = (
            # "[Subject]: {}\n"
            # "[Relation]: {}\n"
            "[Event and News Summary]: {}\n"
            "You need to summarize all the news summaries and events in [Event and News Summary]. Your response should only include the generated summary."
        ).format(",".join(["\n["+ str(event[0]) + ':' + str(event[1])+']' for event in graph_align]))
    return input


def run_llm_prune(prompt,prompt_rules, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # openai.api_key = opeani_api_keys
    client = OpenAI(
        api_key=opeani_api_keys,
        )
    messages = [{"role":"system","content":prompt_rules}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    response = client.chat.completions.create(
                    model=engine,
                    response_format={ "type": "json_object" },
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed = 12345,
                    # top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=0)
    result = response.choices[0].message.content
    return result


def run_llm(prompt,prompt_rules, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # openai.api_key = opeani_api_keys
    client = OpenAI(
        api_key=opeani_api_keys,
        
        )
    messages = [{"role":"system","content":prompt_rules}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    response = client.chat.completions.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed = 12345,
                    # top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=0)
    result = response.choices[0].message.content
    return result


class LinkDataset(object):
    def __init__(self):
        self.date2iddate2id = read_dictionary("./data/date2id.txt")
        with open('./data/candidate_set_for_ceid.json','r') as f:
            self.candidate_set_for_ceid = json.load(f)
        with open('./data/ceid2timid.json', 'r') as file:
            self.ceid2timid = json.load(file)
        with open('./data/doc_clean_ALL.json', 'r') as file:
            self.doc = json.load(file)
        with open('./data/summary_list_clean.json', 'r') as file:
            self.doc_summary = json.load(file)
        # ce_time = sorted(self.ceid2timid[str(row['ce_id'])])
        with open('./data/Md52timid.json','r') as f:
            self.Md52timid = json.load(f)
        self.df_all = pd.read_csv("./data/all_events_final.csv",sep="\t",dtype={"Relation_id": str})
        # self.graph_memory = {}

    def read_dictionary(self, filename):
        d = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                d[int(line[1])] = line[0]
        return d


    def is_first(self, subject):
        return True if self.df_all.loc[self.df_all['Subject'] == subject].shape[0]==0 else False 


    def graph_exploration(self, query_subject, ceid,timid,histlen):
        # row['timid'], row['ce_id'], ce_time, histlen = 1
        # self.graph_memory={}
        event_temporal_exploration = self.df_all.loc[(self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        
        event_spatial_exploration_ce = event_temporal_exploration.loc[(event_temporal_exploration['ce_id'] == ceid)]
        # event_spatial_exploration_sub = event_temporal_exploration.loc[(self.df_all['Subject'] == query_subject)]

        # event_spatial_exploration = pd.concat([event_spatial_exploration_ce, event_spatial_exploration_sub], axis=0)
        # event_spatial_exploration.drop_duplicates(inplace=True)
        entity_list = set(event_spatial_exploration_ce['Subject'].unique().tolist() + event_spatial_exploration_ce['Object'].unique().tolist())
        return entity_list

    def prune_entity(self, subject, relevant_subject_set, args):
        prompt_prune = generate_prune_input(subject, relevant_subject_set)
        prompt_prune_rules = generate_prompt_rules('prune_entity')
        prune_result = run_llm_prune(prompt_prune, prompt_prune_rules, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)
        try:
            prune_result = eval(prune_result)['selected_entities']
        except:
            prune_result = []

        # TODO
        # print(f"prompt_prune:\n{prompt_prune}")
        # print(f"prompt_prune_rules:\n{prompt_prune_rules}")
        # print(prune_result)
        # print("*"*40)
        return prune_result


    def get_truck_text(self, md5, trunk):
        data = self.doc.get(md5, {})
        title = data[0]
        text = data[1]
        # title, text = data.get('Title', ''), data.get('Text', [])
        trunk_parts = trunk.strip('[]').split(':')
        # if int(trunk_parts[2])-int(trunk_parts[1])>=2:
        #     trunk_parts[2] = int(trunk_parts[1])+2
        if trunk_parts[0] == '0':
            return title
        else:
            start, end = int(trunk_parts[1]), int(trunk_parts[2]) + 1
            return ''.join(text[start:end])


    def search_events(self, relevant_subject_set, timid, histlen):
        related_events_subject = self.df_all.loc[(self.df_all['Subject'].isin(relevant_subject_set)) & (self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        related_events_object = self.df_all.loc[(self.df_all['Object'].isin(relevant_subject_set)) & (self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        events = []
        for _, event in related_events_subject.iterrows():
            events.append([event['Subject'], event['Relation_choice'], event['Object'], event['timid']])
        for _, event in related_events_object.iterrows():
            events.append([event['Subject'], event['Relation_choice'], event['Object'], event['timid']])
        events = sorted(events, key=lambda x: x[3], reverse=True)
        events = events[:20]
        return events

    def text_align(self, subject, text_list,args):
        # text_list = [self.get_truck_text(event[1][0],event[1][1]) for event in event_list]

        final_result = []
        
        for i in range(len(text_list)):
            prompt_summary_rules = generate_prompt_rules('text_align')
            prompt_summary = generate_prompt_text(subject, text_list[i][1])
            prune_result = run_llm(prompt_summary, prompt_summary_rules, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)
            # TODO
            # print(prompt_summary_rules)
            # print(prompt_summary)
            # print(prune_result)
            # print('*'*40)
            final_result.append([text_list[i][0],prune_result])
 
        return final_result

    def retrieve_text(self, relevant_events, ceid, timid, histlen):
        event_temporal_exploration = self.df_all.loc[(self.df_all['ce_id'] == ceid) &(self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        Md5_list = event_temporal_exploration['Md5'].unique().tolist()
        documents = []
        for Md5 in Md5_list:
            documents.append(Document(text=''.join(self.doc[Md5][1]),metadata={'timid':self.Md52timid[Md5][0],'Md5':Md5,'title':self.doc[Md5][0]}))
        # print(len(documents))
        llm = llama_index_OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=50)
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=256, chunk_overlap=20)
        print('start')
        index = VectorStoreIndex.from_documents(documents, service_context=service_context,use_async=True)
        # print('done')
        retriever = index.as_retriever(retriever_mode='embedding')
        response = retriever.retrieve(str(relevant_events))

        return response

    def retrieve_text_BM25(self, relevant_events,subject, ceid, timid, histlen):
        event_temporal_exploration = self.df_all.loc[(self.df_all['ce_id'] == ceid) &(self.df_all['timid'] < timid) & (self.df_all['timid'] >= timid-histlen)]
        Md5_list = event_temporal_exploration['Md5'].unique().tolist()
        if len(Md5_list)!=0:
            documents = []
            doc_info = []
            for Md5 in Md5_list:
                documents.append(''.join(self.doc[Md5][1]))
                doc_info.append({'timid':self.Md52timid[Md5][0],'Md5':Md5,'title':self.doc[Md5][0]})

            query = str(" ".join(relevant_events))
            tokenized_query_gt = query.split(" ")
            tokenized_query_t = subject.split(" ")
            tokenized_documents = [document.split(" ") for document in documents]
            bm25_model = BM25Okapi(tokenized_documents)
            # average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())

            scores_gt = bm25_model.get_scores(tokenized_query_gt)
            scores_t = bm25_model.get_scores(tokenized_query_t)
            # idx = scores.index(max(scores))
            max_indices_gt = sorted(enumerate(scores_gt), key=lambda x: x[1], reverse=True)[:10]
            max_indices_t = sorted(enumerate(scores_t), key=lambda x: x[1], reverse=True)[:10]
            response_gt = [[documents[index],doc_info[index]] for index, value in max_indices_gt]
            response_t = [[documents[index],doc_info[index]] for index, value in max_indices_t]
        else:
            response_gt = []
            response_t = []
        return response_gt,response_t


@torch.inference_mode()
def main(args):
    sys.stdout = Logger(stream = sys.stdout)
    # sys.stderr = Logger(stream = sys.stderr)
    print(f"args.data_path:{args.data_path}")
    print(f"args.max_tokens:{args.max_tokens}")
    print(f"args.temperature:{args.temperature}")
    print(f"args.engine:{args.engine}")
    print(f"args.histlen:{args.histlen}")
    

    df = pd.read_csv(args.data_path, sep="\t",dtype={"Relation_id": str})
    df = df.sort_values('timid')
    # df = df.sort_values('timid')

    database = LinkDataset()
    total_num = 0
    true_num_gt = 0
    true_num_t = 0
    true_num_test0 = 0
    true_num_test1 = 0
    true_num_test2 = 0
    true_num_test3 = 0
    # all_result = []
    all_result_gt = []
    all_result_t = []
    current_time=df['timid'].min()
    with tqdm(total=df.shape[0], desc="Event Forecasting") as pbar:
        for index_event, row in df.iterrows():
            # row = df.iloc[1300]
            group_num = row['groupid']
            id = row['ID']
            timid = row['timid']
            histlen = args.histlen
            # histlen = 30

            relevant_subject_candidate_set = database.graph_exploration(row['Subject'], row['ce_id'], row['timid'], histlen)
            if len(relevant_subject_candidate_set) != 0:
                relevant_subject_set = database.prune_entity(row['Subject'], relevant_subject_candidate_set, args)
                relevant_subject_set.append(row['Subject'])
                relevant_events = database.search_events(list(set(relevant_subject_set)), row['timid'], histlen)
            else:
                relevant_events = None

            retrieve_result_gt,retrieve_result_t = database.retrieve_text_BM25(relevant_subject_set,row['Subject'],row['ce_id'],row['timid'],histlen)
            if len(retrieve_result_gt) != 0:
                # text_with_timid_gt = {r.metadata['Md5']:(r.metadata['timid'],r.text)for r in retrieve_result_gt}
                text_with_timid_gt = {r[1]['Md5']:(r[1]['timid'],r[0])for r in retrieve_result_gt}
                text_sort_by_timid_gt = sorted(text_with_timid_gt.values(), key=lambda item: item[0], reverse=True)
                text_sort_by_timid_gt = text_sort_by_timid_gt[0:5]
                complement_summary_gt = database.text_align(row['Subject'],text_sort_by_timid_gt,args)
            else:
                complement_summary_gt=[]
            prompt_reasoning_gt,correct_option_gt,mapped_options_gt = generate_reasoning_input(row, relevant_events, complement_summary_gt,last=True)
            prompt_reasoning_rules_gt = generate_prompt_rules('reasoning')
            result_gt = run_llm(prompt_reasoning_gt, prompt_reasoning_rules_gt, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)

            if len(retrieve_result_t) != 0:
                text_with_timid_t = {r[1]['Md5']:(r[1]['timid'],r[0])for r in retrieve_result_t}
                # text_with_timid_t = {r.metadata['Md5']:(r.metadata['timid'],r.text)for r in retrieve_result_t}
                text_sort_by_timid_t = sorted(text_with_timid_t.values(), key=lambda item: item[0], reverse=True)
                text_sort_by_timid_t = text_sort_by_timid_t[0:5]
                complement_summary_t = database.text_align(row['Subject'],text_sort_by_timid_t,args)
            else:
                complement_summary_t=[]
            prompt_reasoning_t,correct_option_t,mapped_options_t = generate_reasoning_input_t(row, complement_summary_t,last=True)
            prompt_reasoning_rules_t = generate_prompt_rules('reasoning_t')
            result_t = run_llm(prompt_reasoning_t, prompt_reasoning_rules_t, args.temperature, args.max_tokens, args.opeani_api_keys, args.engine)
            
            # TODO
            # print(f"prompt_reasoning_rules_gt:\n{prompt_reasoning_rules_gt}")
            # print(f"prompt_reasoning_gt:\n{prompt_reasoning_gt}")
            # print(f"result_gt:\n{result_gt}")
            # print(f"prompt_reasoning_rules_t:\n{prompt_reasoning_rules_t}")
            # print(f"prompt_reasoning_t:\n{prompt_reasoning_t}")
            # print(f"result_t:\n{result_t}")
            response_gt = parse_outputs(result_gt)
            response_t = parse_outputs(result_t)

            total_num += 1
            pbar.update(1)
            if any(item in correct_option_gt for item in response_gt):
                true_num_gt +=1
            if any(item in correct_option_t for item in response_t):
                true_num_t +=1
    
            all_result_gt.append(
                {
                    "correct_option": correct_option_gt,
                    # "relevant_subject_set": relevant_subject_set,
                    "predict": result_gt,
                    "mapped_options": mapped_options_gt,
                    "group_num":group_num,
                    "ID": id,
                    "index": index_event
                }
            )

            all_result_t.append(
                {
                    "correct_option": correct_option_t,
                    # "relevant_subject_set": relevant_subject_set,
                    "predict": result_t,
                    "mapped_options": mapped_options_t,
                    "group_num":group_num,
                    "ID": id,
                    "index": index_event
                }
            )


        group_dict = df['groupid'].value_counts().to_dict()
        precision_gt = true_num_gt/total_num
        precision_t = true_num_t/total_num

        print(f"total_num:{total_num} == {sum(list(group_dict.values()))}")
        print(f'precision:{precision_gt}')
        print(f'precision:{precision_t}')

        with open(f"./results/gpt-3.5-turbo/retriever/result_with_history_retrieve_graph_text_ce_BM25.json", "w") as f: 
            json.dump(all_result_gt, f, ensure_ascii=False, indent=2)
        with open(f"./results/gpt-3.5-turbo/retriever/result_with_history_retrieve_text_ce_BM25.json", "w") as f: 
            json.dump(all_result_t, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add_model_args(parser)
    parser.add_argument("--data_path", type=str, default="all_events_final_test.csv")
    parser.add_argument("--opeani_api_keys", type=str, default="OPENAI KEY")
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--histlen", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    main(args)



    


