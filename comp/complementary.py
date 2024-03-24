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
        "1. Extract image information as a sub-event, rather than multiple sub-events.",
        "2. Based on key information or sub-events in the text, determine the theme or focus of the image.",
        "3. Directly relate the extracted image information to the relevant news event, enhancing understanding of the news content.",
        "4. Prioritize and emphasize the most newsworthy and important details in the image, such as specific actions, emotions, or thematic features.",
        "5. Ensure that the extracted information directly originates from the provided image and news article, avoiding the introduction of fictional content or speculative details.",
        "6. Summarize the information concisely and clearly, using plain language. Avoid excessive detail or subjective commentary.",
        "7. Maintain an objective and impartial attitude when describing the image, avoiding the insertion of personal viewpoints or interpretations.\n"]

    format_ = [
        "Image Summary: [A summary of the key information or sub-event for each image, without including any prefix phrases]"]
     
    prompt_rules = f'You are a professional news writer. Please extract image information based on the provided news content, following the rules below and incorporating your own domain knowledge:\n' + '\n'.join(
        rules) + f'\nPlease present the extracted image information in the following format:\n' + '\n'.join(format_)

    return prompt_rules


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DIVICES'] = '0'

    #dic_csv = read_csv()  # transfer extraction results to json, load in dic_csv
    prompt = get_prompt_rules()

    print('Prompt:')
    print(prompt)

    genai.configure(api_key="AIzaSyCwlDwkcKdUc0oXraTAukwpi4AgU0_C-bc", transport="rest")

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

    doc_sum_list = json.load(open(f'C:\\Users\\Administrator\\Documents\\WeChat Files\\wxid_4nnjq4841k2s12\\FileStorage\\File\\2024-03\\aligned_compl\\summary_gemini_wo_event{split_number}_refine.json', 'r'))
    md5_split = json.load(open('md5_split.json', 'r'))

    #doc_pho = json.load(open('/mnt/hdd/hxli/Datasets/Dataset_haoxuan/image_3/doc2pho.json', 'r'))

    md5_classification = json.load(open(f'C:\\Users\Administrator\\Documents\\WeChat Files\\wxid_4nnjq4841k2s12\\FileStorage\\File\\2024-03\\aligned_compl\\classification_gemini_{split_number}_refine.json', 'r'))

    md5_wrong = {}
    md5_comp = {}
    cnt = 0

    for doc_md5 in md5_split[f'{split_number}']:


        print()
        print('-----------------------------------')
        print(f"Md5: {doc_md5}")
        doc_sum_i = doc_sum_list[doc_md5]
        doc_sum_i = doc_sum_i[1:]

        doc_content = "* " + "* ".join(doc_sum_i)
        #print(doc_content)

        pho_list = md5_classification[doc_md5]

        md5_comp[doc_md5] = {}

        for pho_list_i in pho_list.keys():
            img_path = "D:\\edge download\\MIDEAST_MUTISOURCE\\MIDEAST_MUTISOURCE\\images\\" + f"{pho_list_i}" + ".jpg"

            if pho_list[pho_list_i] == 'complementary':
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
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                images = [image]

                prompt_parts = [f"{prompt}\nPlease extract image information following the above guidelines and present it in the described format:\n"
                                f"Image: ",
                                images[0],
                                f".\nThe sub-events of news article:\n{doc_content}\n\nThe information extracted from the image:\n",]
                
                print(f"\nImage: {pho_list_i}.\nThe sub-events of news article:\n{doc_content}\n\nPlease extract image information following the above guidelines and present it in the described format\n")

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
                
                output_i = response.text
                output_i = output_i.strip()
                output_i = output_i.split('\n')
                for ii, output_i_i in enumerate(output_i):
                    output_i[ii] = "* " + output_i_i
                md5_comp[doc_md5][pho_list_i] = output_i
                print(md5_comp[doc_md5][pho_list_i])


                sleep(1)
                print()

                print(cnt)
            cnt += 1
        if cnt >= 21:
            break

        print('-----------------------------------')
        print()

        
    
    json.dump(md5_wrong, open(f'complementary_md5_failed_{split_number}.json', 'w'))
    json.dump(md5_comp, open(f'complementary_gemini_{split_number}_2.json', 'w'))