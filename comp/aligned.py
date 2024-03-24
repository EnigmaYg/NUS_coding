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
        "1. For the final judgement, please answer with the serial number of the sub-event. For example: [The number of the sub-event most relevant to the image is 1.]",
        "2. Identify the main subjects or objects prominently featured in the image. Sub-events that provide details, background information or context directly about these central visual elements are highly relevant.",
        "3. If people are depicted, identify who those individuals are. Sub-events involving those particular people should take priority.",
        "4. Analyze the overall activities, actions, emotions or mood being portrayed in the image. Relevant sub-events likely delve into similar situations, occurrences or sentiments illustrated.",
        "5. Take note of the specific location, setting or environment depicted in the image. Prioritize sub-events that discuss that geographic area, type of place, or related events.",
        "6. Look for any text, logos, labeled items or signs visible in the image content. Sub-events elaborating on the organizations, companies, products or public figures represented by those texts are applicable.\n",]
     
    prompt_rules = f'You are a professional news writer.\nPlease determine which sub-event in the news the image is most relevant to based on the following rules and your own domain knowledge:\n' + '\n'.join(
        rules)                     

    return prompt_rules


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DIVICES'] = '0'

    #dic_csv = read_csv()  # transfer extraction results to json, load in dic_csv
    prompt = get_prompt_rules()

    print('Prompt:')
    print(prompt)

    genai.configure(api_key="AIzaSyDsdAcoz6IGiFmImGFGgkvUQoXSYZHZyMg")

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
    

    split_number = 6

    doc_sum_list = json.load(open(f'/storage_fast/ccye/zmyang/20240317/summary_gemini_wo_event{split_number}_refine.json', 'r'))
    md5_split = json.load(open('/storage_fast/ccye/zmyang/20240317/md5_split.json', 'r'))

    #doc_pho = json.load(open('/mnt/hdd/hxli/Datasets/Dataset_haoxuan/image_3/doc2pho.json', 'r'))

    md5_classification = json.load(open(f'/storage_fast/ccye/zmyang/20240317/classification_gemini_{split_number}_refine.json', 'r'))

    md5_wrong = {}
    md5_aligned = {}

    for doc_md5 in md5_split[f'{split_number}']:

        print()
        print('-----------------------------------')
        print(f"Md5: {doc_md5}")
        doc_sum_i = doc_sum_list[doc_md5]
        doc_sum_i = doc_sum_i[1:]

        doc_content = ""
        num_sum = len(doc_sum_i)
        for num in range(len(doc_sum_i)):
            doc_content = doc_content + f"{num+1}. {doc_sum_i[num]}"
        #print(doc_content)

        pho_list = md5_classification[doc_md5]

        md5_aligned[doc_md5] = {}

        for pho_list_i in pho_list.keys():
            img_path = "/storage_fast/ccye/zmyang/20240316/MIDEAST_MUTISOURCE/images/" + f"{pho_list_i}" + ".jpg"

            if pho_list[pho_list_i] == 'aligned':
                image = Image.open(img_path)

                '''if not os.path.isfile(img_path):
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
                    continue'''
                
                if image.mode == 'P':
                    image = image.convert('RGB')
                images = [image]

                prompt_parts = [f"{prompt}\nImage: ",
                                images[0],
                                f".\nNews article:\n{doc_content}\n\nThe number of the sub-event most relevant to the image is ",]
                
                print(f"\nImage: {pho_list_i}.\nThe sub-events of news article:\n{doc_content}\n\nThe number of the sub-event most relevant to the image is ", end="")

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

                        image_i = images[0]
                        width_i, height_i = image_i.size
                        image_i = image_i.resize((width_i//2, height_i//2))


                        images = [image_i]

                        prompt_parts = [f"{prompt}\nImage: ",
                            images[0],
                            f".\nNews article:\n{doc_content}\n\nThe relationship between this image and the news article is",]
                        always_i = always_i + 1
                        sleep(5)
                        continue
                    except google.api_core.exceptions.DeadlineExceeded:
                        print('4', end='')
                        always_i = always_i + 1
                        sleep(5)
                        continue
                    
                    try :
                        print('\n' + response.text)

                        judge = False
                        for num_i in range(num_sum):
                            if f"{num_i+1}." in response.text:
                                judge = True
                                break
                        if judge:
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
                
                for num_i in range(num_sum):
                    if f"{num_i+1}." in response.text:
                        md5_aligned[doc_md5][pho_list_i] = num_i+1
                        break


                sleep(1)
                print()

        print('-----------------------------------')
        print()

    
    json.dump(md5_wrong, open(f'aligned_md5_failed_{split_number}.json', 'w'))
    json.dump(md5_aligned, open(f'aligned_gemini_{split_number}.json', 'w'))