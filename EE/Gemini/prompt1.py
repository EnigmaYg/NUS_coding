import pathlib
import textwrap
import json
import os
import time
from tqdm import tqdm

import google.generativeai as genai

from IPython.display import Markdown

GOOGLE_API_KEY = 'AIzaSyCb3hB7lA7mVuryKE2VVQgU00PC5UINGdU'
current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def get_prompt1_relation(article, md5):
    file_in_parent_directory = os.path.join(parent_directory, 'dict_id2ont_choice.json')
    dict_id2ont = json.load(open(file_in_parent_directory, 'r'))

    first_ont_list = []
    ont_list_description = []
    for i in range(1, 21):
        curr_id = str(i) + "0" if i > 9 else "0" + str(i) + "0"
        relation = dict_id2ont[curr_id]["choice"] + "||" + dict_id2ont[curr_id]["description"]
        first_ont_list.append(dict_id2ont[curr_id]["choice"])
        ont_list_description.append(relation)
        # first_ont_list.append(dict_id2ont[curr_id]["choice"])
    rules = [
        "1. Extract each event in format: {[event actor one]; [event relation]; [event actor two]}.",
        " 2. The term [event actor one] and [event actor two] should be concrete and real-world entities, countries or international organizations.",
        " For example, in following text: 'Canada's Opposition To Palestine's United Nations Involvement Appalling, Says Envoy.', [event actor one] should be 'Canada', [event actor two] should be 'Palestine'.",
        " 3. Only choose [event relation] from this relation candidate list: " + ', '.join(first_ont_list) + ',',
        " 4. Here is the description list for relation candidate: " + ', '.join(ont_list_description) + ',' + "each item in the preceding list represents an option,",
        " before '||' is relation you could choose, after '||' is description for that relation.",
        " 5. Only extract events that have happened or is happening, and not extract future events.",
        " 6. Output needs to be in JSON format. The keys are the identifiers for each trunk segment, and the values are the events extracted from the respective trunk segment."
        " Identifiers for each trunk segment is in format of [x], x is an integer."
    ]
    prompt_rules = f'You are an assistant to perform structured event extraction from news articles that were splited into several trunks with following rules:\n' + '\n'.join(
        rules)
    # "3.If an actor involves multiple objects, please split it into multiple actors, each forming a separate event."
    # "For example, 'United States, Britain, and Russia; Consult or meet; China' should be split into 'Russia; Consult or meet; China,'"
    # "'Britain; Consult or meet; China,' and 'United States; Consult or meet; China,' and output them separately."
    prompt_example = "For example, given the example article in several trunks:\n" + \
                     "[0] Egypt committed to boosting economic cooperation with Lebanon\n" + \
                     "[1](MENAFN- Daily News Egypt) Egypt is committed to enforcing economic cooperation with Lebanon, President Abdel Fattah Al-Sisi said during his meeting with Lebanese parliamentary speaker Nabih Berri.\n" + \
                     "\nList all events by rules, the extraction result of the example is:\n" + \
                     "{\n" \
                     '  "[0]" :\n' \
                     '    ["Egypt; Express intent to cooperate; Lebanon"], \n' \
                     '  "[1]" :\n' \
                     '    ["Egypt president Abdel Fattah Al-Sisi; Consult or meet; Lebanese parliamentary speaker Nabih Berri | Lebanese parliamentary speaker Nabih Berri; Consult or meet; Egypt president Abdel Fattah Al-Sisi"]\n' \
                     '}'
    raw_tokens = 0
    raw_tokens_title = article['Title'].split(' ')
    raw_tokens_title_length = len(raw_tokens_title)
    raw_tokens += raw_tokens_title_length
    text_article = article['Text']
    i = 0
    for i in range(1, len(text_article) + 1):
        raw_tokens_text = str(text_article[:i]).split(' ')
        raw_tokens_text_length = len(raw_tokens_text)
        raw_tokens += raw_tokens_text_length
        if raw_tokens > 2048:
            i -= 1
            break
        if i != len(text_article):
            raw_tokens -= raw_tokens_text_length
    text = get_trunk(article, i, md5)
    # 1024 tokens, by sentence
    prompt_instruction = 'Now, given the query article:\n' + text + '\n\nList all events by rules, the extraction result of the query article is:'

    return '\n'.join([prompt_rules, prompt_example, prompt_instruction])

def get_trunk(article, i, md5):
    text = []
    text.append("[0]" + str(article['Title']))
    trunk_id = json.load(open('results/trunk_id.json', 'r'))
    x = 0
    cnt = 1
    id = ["[0]/[0:0:0]"]
    for j in range(1, i+1):
        if abs(150 - len(str(article['Text'][x:j]).split(' '))) <= 50:
            trunk = "[" + str(cnt) + "] "
            id.append("[" + str(cnt) + "]" + "/" + "[" + str(cnt) + ":" + str(x) + ":" + str(j-1) + "]")
            text.append(trunk + "".join(article['Text'][x:j]))
            x = j
            cnt += 1
    if x != j:
        trunk = "[" + str(cnt) + "] "
        id.append("[" + str(cnt) + "]" + "/" + "[" + str(cnt) + ":" + str(x) + ":" + str(j - 1) + "]")
        text.append(trunk + "".join(article['Text'][x:j]))
    trunk_id[md5] = id
    with open('results/trunk_id.json', 'w') as file:
        json.dump(trunk_id, file, indent=2)
    return '\n'.join(text)

if __name__ == '__main__':
    genai.configure(api_key=GOOGLE_API_KEY,transport='rest')

    '''for m in genai.list_models():
      if 'generateContent' in m.supported_generation_methods:
        print(m.name)'''

    model = genai.GenerativeModel('gemini-pro')

    cnt = 0

    file_in_parent_directory = os.path.join(parent_directory, 'md5_list.json')
    mread = json.load(open(file_in_parent_directory, 'r'))
    file_in_parent_directory = os.path.join(parent_directory, 'MIDEAST_doc_v2.json')
    doc_list = json.load(open(file_in_parent_directory, 'r'))
    md5_list = mread['md5']
    start_idx = 0
    end_idx = 0
    if end_idx == 0 or end_idx > len(md5_list) - 1:
        end_idx = len(md5_list) - 1
    exists_results = os.listdir('results/raw/level1/')
    for idx, md5 in tqdm(enumerate(md5_list), total=len(md5_list)):
        if idx < start_idx or idx > end_idx:
            continue

        if md5 + '.json' in exists_results:
            continue
        try:
            # print(md5)
            md5 = 'ad4085a49927e5a004924e55eecb3147'
            article = doc_list[md5]

            msg = get_prompt1_relation(article, md5)

            # print(msg)
            response = model.generate_content(msg)
            print(response.prompt_feedback)
            to_markdown(response.text)

            # print(response.text)
            with open('results/raw/level1/' + md5 + '.json', 'w') as fresult:
                fresult.write(response.text)
            cnt += 1

        except Exception as e:
            with open('results/raw/errors/' + md5, 'a') as ferror:
                ferror.write('\n---\n')
                ferror.write(str(e))
                print(cnt)
                print(md5)
    print(cnt)
