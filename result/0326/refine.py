import json

split_number = 7

raw = json.load(open(f'classification_gemini_graph_{split_number}.json', 'r'))
fail = json.load(open(f'md5_failed_{split_number}.json', 'r'))

for key, value in fail.items():
    for v in value:
        string_value = "".join(v)
        raw[key][string_value] = 'safety'

json.dump(raw, open(f'classification_gemini_graph_{split_number}_refine.json', 'w'), indent=4)