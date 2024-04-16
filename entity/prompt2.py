import openai
import json
import os
import textwrap
import time
from tqdm import tqdm

import google.generativeai as genai

from IPython.display import Markdown

API_KEY = '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'
openai.api_key = API_KEY

GOOGLE_API_KEY = 'AIzaSyBc-G2nkaqvvMoc3Co3vw3gQs2e8YNY4dE'
current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def get_prompt1_relation(article, md5):
    rules = [
        "1. Extract entities in format: {[entity one] || [entity two] || [entity three] and so on}.",
        " 2. The entities should be concrete and real-world, for example countries, international organizations or political figures.",
        " 3. All entities need to have an exact presence in the original article.",
        " 4. Output each entity only once, avoiding duplication.",
        " 5. Separate all entities with '||' when outputting.\n"
    ]
    prompt_rules = f'You are an assistant to perform entities extraction from news articles with following rules:\n' + '\n'.join(
        rules)
    # "3.If an actor involves multiple objects, please split it into multiple actors, each forming a separate event."
    # "For example, 'United States, Britain, and Russia; Consult or meet; China' should be split into 'Russia; Consult or meet; China,'"
    # "'Britain; Consult or meet; China,' and 'United States; Consult or meet; China,' and output them separately."
    prompt_example = "For example, given the example article:\n" + \
                     "'Canada's Opposition To Palestine's United Nations Involvement Appalling, Says Envoy.'\n" + \
                     "List all entities by rules, the extraction result of the example is:\n" + \
                     "Canada || Palestine || United Nations || Envoy\n"

    # 1024 tokens, by sentence
    prompt_instruction = 'Now, given the query article:\n' + '\n'.join(article["Text"]) + '\n\nList all entities by rules, the extraction result of the query article is:'

    return '\n'.join([prompt_rules, prompt_example, prompt_instruction])


def get_results():
    cnt = 0
    file_in_parent_directory = os.path.join(parent_directory, 'md5_list.json')
    mread = json.load(open(file_in_parent_directory, 'r'))
    file_in_parent_directory = os.path.join(parent_directory, 'MIDEAST_doc_v2.json')
    doc_list = json.load(open(file_in_parent_directory, 'r'))
    md5_list = mread['md5']
    genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

    '''for m in genai.list_models():
      if 'generateContent' in m.supported_generation_methods:
        print(m.name)'''

    model = genai.GenerativeModel('gemini-pro')
    start_idx = 0
    end_idx = 0
    if end_idx == 0 or end_idx > len(md5_list) - 1:
        end_idx = len(md5_list) - 1
    exists_results = os.listdir('Gemini_results/raw/')
    for idx, md5 in tqdm(enumerate(md5_list), total=len(md5_list)):
        if idx < start_idx or idx > end_idx:
            continue

        if md5 in exists_results:
            continue
        try:
            # print(md5)
            article = doc_list[md5]
            msg = get_prompt1_relation(article, md5)
            # print(msg)
            response = model.generate_content(msg)
            # print(response.prompt_feedback)
            to_markdown(response.text)
            print(response.text)

            with open('Gemini_results/raw/' + md5, 'w') as fresult:
                fresult.write(response.text)
            cnt += 1

        except Exception as e:
            with open('Gemini_results/errors/' + md5, 'a') as ferror:
                ferror.write('\n---\n')
                ferror.write(str(e))
                print(cnt)
                break
    print(cnt)

if __name__ == '__main__':
    get_results()