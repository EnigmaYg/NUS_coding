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
        "2. The relationship between an image and a news article is aligned meaning that the information in the image is mentioned in the news article, without requiring all the content of the news to be depicted in the image. As long as the specific events covered in the image are included in the news, it should be considered aligned.",
        "3. Please note that the photo may not provide specific information about the objects, such as the nationality of the soldiers or the armed group they belong to. However, as long as the news article mentions specific information related to the events depicted in the photo, even if this information is not clearly displayed in the picture, we still consider it aligned."]

    prompt_rules = f'You are a professional news writer.\nPlease evaluate whether the relationship between images and news aligns based on the following rules and your own domain knowledge:\n' + '\n'.join(
        rules)
    return prompt_rules


if __name__ == '__main__':
    json_file = json.load(open('/storage_fast/ccye/zmyang/20240408/text_alin.json', 'r'))
    doc = json.load(open(f'/storage_fast/ccye/zmyang/MM_photo/MIDEAST_doc_v2.json', 'r'))
    doc_sum = json.load(open(f'/storage_fast/ccye/zmyang/20240408/summary_all.json', 'r'))
    prompt = get_prompt()
    print('aligned_text_result')
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
        doc_con_i = doc[key]["Text"]
        doc_content = " ".join(doc_con_i)

        doc_sum_i = doc_sum[key]
        doc_sum_i = doc_sum_i[1:]
        doc_summary = "* " + "* ".join(doc_sum_i)

        image_path = "/storage_fast/ccye/zmyang/20240316/MIDEAST_MUTISOURCE/images/" + f"{value}" +".jpg"

        prompt_parts1 = f"{prompt}\nNews article:\n{doc_content}\nImage: "
        prompt_parts2 = f".\nThese news summaries may help you evaluating:\n{doc_summary}\n\nIs the relationship between the photos and the news events aligned?"
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

    json.dump(judge, open('align_text_result.json', 'w'))
    print(f'true num is {true}')
    print(f'false num is {false}')
    print(f'null num is {null}')
    print(f'precision: {true / 100}')