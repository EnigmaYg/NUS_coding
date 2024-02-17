import json

with open('doc_clean_ALL.json', 'r', encoding='utf-8') as file:
    doc = json.load(file)

md5 = {"md5": []}
for key, value in doc.items():
    md5["md5"].append(key)

with open("md5_list.json", 'w') as file:
    json.dump(md5, file, indent=2)