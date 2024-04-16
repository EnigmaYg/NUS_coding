import json

num = 7

refine_raw = json.load(open(f'complementary_gemini_{num}_v0.json', 'r'))

store = {}

for key, value in refine_raw.items():
    if value != {}:
        for v_key, v_value in value.items():
            string = ''
            flag = 0
            for item in v_value:
                if 'Theme/Focus:' in item:
                    key_info = item.split("Theme/Focus:")[1].strip()
                    string += key_info
                    flag += 1
                if 'Key Information/Sub-event:' in item:
                    flag += 1
                    key_info = item.split("Key Information/Sub-event:")[1]
                    string += key_info
            if key not in store:
                store[key] = {v_key: string}
            else:
                store[key].update({v_key: string})
    else:
        store[key] = ''

json.dump(store, open(f'complementary_gemini_{num}_refine.json', 'w'), indent=4)
