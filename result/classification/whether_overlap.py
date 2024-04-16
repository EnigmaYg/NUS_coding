import pandas as pd
from tqdm import tqdm
import os
import json
from time import sleep

import google.generativeai as genai
import google.api_core

import PIL
from PIL import Image

def get_prompt_rules():
    rules = [
        "1. Final judgment please choose between [aligned, complementary, irrelevant].",
        "2. The relationship between an image and a news article is aligned if the image's subject matter and depicted event are highly related to the news and the specific event shown in the image is already mentioned in detail in the article's description.",
        "3. The relationship between an image and a news article is complementary if the image's overall theme and background information are highly related to the news, but the specific event depicted in the image is not mentioned in detail in the article, and the visual information in the image can complement the news story as a whole.",
        "4. Except in cases where the relationship is aligned or complementary, in other cases, the relationship between the image and the text is irrelevant.\n",]
     
    prompt_rules = f'You are a professional news writer.\nPlease judge the relationship between images and news based on the following rules and your own domain knowledge:\n' + '\n'.join(
        rules)                     

    return prompt_rules


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DIVICES'] = '0'

    #dic_csv = read_csv()  # transfer extraction results to json, load in dic_csv
    prompt = get_prompt_rules()

    print('Prompt:')
    print(prompt)

    genai.configure(api_key="AIzaSyADnd7nqSgvz230GsmC_RbcigXRPhtSJz4")

    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
        }
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
        ]
    
    model = genai.GenerativeModel(model_name="gemini-1.0-pro-vision-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    

    split_number = 0

    doc_sum_list = json.load(open(f'/mnt/hdd/hxli/Datasets/Dataset_haoxuan/summary_changhe_gemini/gemini_sum_result/summary_gemini_wo_event{split_number}_refine.json', 'r'))
    md5_split = json.load(open('/mnt/hdd/hxli/Datasets/Dataset_haoxuan/summary_changhe_gemini/md5_split.json', 'r'))

    doc_pho = json.load(open('/mnt/hdd/hxli/Datasets/Dataset_haoxuan/image_3/doc2pho.json', 'r'))

    md5_wrong = {}
    md5_classification = {}
    for doc_md5 in md5_split[f'{split_number}']:

        print()
        print('-----------------------------------')
        print(f"Md5: {doc_md5}")
        doc_sum_i = doc_sum_list[doc_md5]
        doc_sum_i = doc_sum_i[1:]

        doc_content = "* " + "* ".join(doc_sum_i)
        #print(doc_content)

        pho_list = doc_pho[doc_md5]

        md5_classification[doc_md5] = {}

        for pho_list_i in pho_list:
            img_path = "/mnt/hdd/hxli/Datasets/Dataset_haoxuan/image_3/images/" + f"{pho_list_i}" + ".jpg"

            if not os.path.isfile(img_path):
                md5_classification[doc_md5][pho_list_i] = 'NULL'
                print(f"\nImage: {pho_list_i}.")
                print("No image.")
                continue
            
            try:
                image = Image.open(img_path)
            except PIL.UnidentifiedImageError:
                md5_classification[doc_md5][pho_list_i] = 'Wrong'
                print(f"\nImage: {pho_list_i}.")
                print("Can't open image.")
                continue

            if image.mode == 'P':
                image = image.convert('RGB')
            images = [image]

            prompt_parts = [f"{prompt}\nImage: ",
                            images[0],
                            f".\nNews article:\n{doc_content}\n\nThe relationship between this image and the news article is",]
            
            print(f"\nImage: {pho_list_i}.\nNews article:\n{doc_content}\n\nThe relationship between this image and the news article is")
            
            always_i = 0
            while True:
                if always_i == 100:
                    break

                try:
                    response = model.generate_content(prompt_parts)
                except google.api_core.exceptions.InternalServerError:
                        print('1', end='')
                        always_i = always_i + 1
                        sleep(5)
                        continue
                except google.api_core.exceptions.InvalidArgument:   # the image is too big
                        print('3', end='')
                        always_i = 100
                        sleep(5)
                        continue
                except google.api_core.exceptions.DeadlineExceeded:
                        print('4', end='')
                        always_i = always_i + 1
                        sleep(5)
                        continue

                try :
                    print('\n' + response.text)

                    if "aligned" in response.text or "complementary" in response.text or "irrelevant" in response.text:
                        break
                except ValueError:
                    print('2', end='')
                    always_i = always_i + 1
                    sleep(5)
                    continue

            #print(input_text, end='')
            #print(response.text)
            
            if always_i == 100:
                print('Failed')
                if doc_md5 not in md5_wrong:
                    md5_wrong[doc_md5] = [pho_list_i]
                else:
                    md5_wrong[doc_md5].append(pho_list_i)
                
                sleep(1)
                print()
                continue

            if "aligned" in response.text:
                md5_classification[doc_md5][pho_list_i] = "aligned"
            elif "complementary" in response.text:
                md5_classification[doc_md5][pho_list_i] = "complementary"
            elif "irrelevant" in response.text:
                md5_classification[doc_md5][pho_list_i] = "irrelevant"


            sleep(5)
            print()

        print('-----------------------------------')
        print()

        
    
    json.dump(md5_wrong, open(f'/mnt/hdd/hxli/MM_Event_forecasting/gemini/overlap/md5_failed_{split_number}.json', 'w'))
    json.dump(md5_classification, open(f'/mnt/hdd/hxli/MM_Event_forecasting/gemini/overlap/classification_gemini_{split_number}.json', 'w'))