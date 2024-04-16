import openai
import json
import os
from tqdm import tqdm
import base64
import requests

API_KEY = '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'
openai.api_key = API_KEY

# model_id = 'gpt-4-vision-preview'


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_prompt():
    rules = [
        "1. Final evaluation please choose between [Ture, False].",
        "2. The relationship between an image and a news article is complementary meaning that if the image's overall theme and background information are highly related to the news, but the specific event depicted in the image is not mentioned in detail in the article, and the visual information in the image can complement the news story as a whole.",
    ]

    prompt_rules = f'You are a professional news writer.\nPlease evaluate whether the relationship between images and news is complementary based on the following rules and your own domain knowledge:\n' + '\n'.join(
        rules)
    return prompt_rules

if __name__ == '__main__':
    json_file = json.load(open('/storage_fast/ccye/zmyang/20240408/graph_comp.json', 'r'))
    doc = json.load(open(f'/storage_fast/ccye/zmyang/MM_photo/MIDEAST_doc_v2.json', 'r'))
    doc_sum = json.load(open(f'/storage_fast/ccye/zmyang/20240408/summary_all.json', 'r'))
    doc_event = json.load(open(f'/storage_fast/ccye/zmyang/20240326/md5_event.json', 'r'))
    prompt = get_prompt()
    print('complementary_graph_result')
    print("**************************")
    print('Prompt:')
    print(prompt)
    cnt = 0
    judge = {}
    true = 0
    false = 0
    null = 0
    for key, value in json_file.items():
        judge[key] = {}
        print()
        print('-----------------------------------')
        print(f"Md5: {key}")
        doc_sum_i = doc[key]["Text"]
        doc_content = " ".join(doc_sum_i)

        doc_event_list = doc_event[key]["set"]
        doc_event_content = "* " + "\n* ".join(doc_event_list)

        image_path = "/storage_fast/ccye/zmyang/20240316/MIDEAST_MUTISOURCE/images/" + f"{value}" +".jpg"

        prompt_parts1 = f"{prompt}\nNews article:\n{doc_content}\nImage: "
        prompt_parts2 = f".\nNews events:\n{doc_event_content}\n\nIs the relationship between the photos and the news events complementary?"
        try:
            base64_image = encode_image(image_path)
            print("Image Ok.")
            print(f"\nImage: {value}.")
        except:
            print("Image Fail.")
            print(f"\nImage: {value}.")
            continue
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_parts1
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_parts2
                        },
                    ]
                }
            ],
            "temperature": 0.4,
            "top_p": 1,
            "max_tokens": 300
        }
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            print('-----------------------------------')
            # print(response.json().choices[0].message.content)
            response = response.json()
            print(response['choices'][0]['message']['content'])
            print('-----------------------------------\n\n')
        except Exception as e:
            print('response fail')
            print(e)
            continue

        if 'True' in response['choices'][0]['message']['content']:
            judge[key][value] = 'true'
            true += 1
            continue
        elif 'False' in response['choices'][0]['message']['content']:
            judge[key][value] = 'false'
            false += 1
            continue
        else:
            judge[key][value] = 'null'
            null += 1

    json.dump(judge, open('comp_graph_result.json', 'w'))
    print(f'true num is {true}')
    print(f'false num is {false}')
    print(f'null num is {null}')
    print(f'precision: {true/100}')
