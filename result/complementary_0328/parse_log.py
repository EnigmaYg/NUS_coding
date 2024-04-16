import re
import json


md5_classification = {}

def parse_logs(logfile):
    # 读取日志文件内容
    with open(logfile, 'r') as file:
        logs = file.read()

    # 分割日志文本为每段内容
    log_sections = re.split(r'-{9,}', logs)



    for section in log_sections:
        # 提取Md5值
        md5_match = re.search(r'Doc_md5:\s*([a-fA-F0-9]+)', section)
        if md5_match:
            md5_value = md5_match.group(1)
        else:
            md5_value = None
        image_match = re.search(r'image_md5:\s*([a-fA-F0-9]+)', section)
        if image_match:
            image_value = image_match.group(1)
        else:
            image_value = None
        md5_classification[md5_value] = image_value



# 日志文件路径
logfile = 'test_all.log'

# 解析日志文件
parsed_logs = parse_logs(logfile)



json.dump(md5_classification, open(f'complementary_lost.json', 'w'), indent=4)
