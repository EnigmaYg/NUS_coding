import re
import json

def parse_logs(logfile):
    # 读取日志文件内容
    with open(logfile, 'r') as file:
        logs = file.read()

    # 分割日志文本为每段内容
    log_sections = re.split(r'-{20,}', logs)

    parsed_data = []

    for section in log_sections:
        # 提取Md5值
        md5_match = re.search(r'Md5:\s*([a-fA-F0-9]+)', section)
        if md5_match:
            md5_value = md5_match.group(1)
        else:
            md5_value = None

        # 提取所有Image值
        image_values = re.findall(r'Image:\s*([a-fA-F0-9]+)', section)

        # 提取与图像相关的描述信息
        image_sections = re.split(r'Image:\s*', section)
        image_sections = image_sections[1:]
        descriptions = []
        for image_section in image_sections:
            # 提取与图像相关的描述信息
            description_matches = re.findall(r'(complementary|aligned|irrelevant|failed)', image_section)
            unique_descriptions = set(description_matches)
            if 'complementary' or 'aligned' or 'irrelevant' in unique_descriptions:
                if 'failed' in unique_descriptions:
                    unique_descriptions.remove('failed')
            str_description = ', '.join(unique_descriptions)
            descriptions.append(str_description)

        parsed_data.append({'Md5': md5_value, 'Images': image_values, 'Descriptions': descriptions})

    return parsed_data

# 日志文件路径
logfile = 'test_new_11.log'

# 解析日志文件
parsed_logs = parse_logs(logfile)


md5_wrong = {}
md5_classification = {}
for log in parsed_logs:
    if log['Md5'] != None:
        cnt = 0
        md5_classification[log['Md5']] = {}
        for image in log['Images']:
            if 'complementary' == log['Descriptions'][cnt] or 'aligned' == log['Descriptions'][cnt] or 'irrelevant' == log['Descriptions'][cnt]:
                print(log['Descriptions'][cnt])
                md5_classification[log['Md5']][image] = log['Descriptions'][cnt]
            elif log['Descriptions'][cnt] == '':
                if log['Md5'] not in md5_wrong:
                    md5_wrong[log['Md5']] = [image]
                else:
                    md5_wrong[log['Md5']].append(image)
            cnt += 1

split_number = 1
json.dump(md5_wrong, open(f'md5_failed_{split_number}_ll.json', 'w'), indent=4)
json.dump(md5_classification, open(f'classification_gemini_graph_{split_number}_ll.json', 'w'), indent=4)
