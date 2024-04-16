import json

split_number = 1

raw1 = json.load(open(f'classification_gemini_graph_{split_number}.json', 'r'))
raw2 = json.load(open(f'classification_gemini_graph_{split_number}_l.json', 'r'))
raw3 = json.load(open(f'classification_gemini_graph_{split_number}_ll.json', 'r'))
fail1 = json.load(open(f'md5_failed_{split_number}.json', 'r'))
fail2 = json.load(open(f'md5_failed_{split_number}_l.json', 'r'))
fail3 = json.load(open(f'md5_failed_{split_number}_ll.json', 'r'))

raw4 = {}
for key, value in raw1.items():
    raw4[key] = value
for key, value in raw2.items():
    raw4[key] = value
for key, value in raw3.items():
    raw4[key] = value
for key, value in fail1.items():
    for v in value:
        string_value = "".join(v)
        raw4[key][string_value] = 'safety'
for key, value in fail2.items():
    for v in value:
        string_value = "".join(v)
        raw4[key][string_value] = 'safety'
for key, value in fail3.items():
    for v in value:
        string_value = "".join(v)
        raw4[key][string_value] = 'safety'

json.dump(raw4, open(f'classification_gemini_graph_{split_number}_refine.json', 'w'), indent=4)