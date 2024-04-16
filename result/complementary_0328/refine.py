import json

num = 7

raw = json.load(open(f'complementary_gemini_graph_all.json', 'r'))
add_item = json.load(open(f'complementary_lost_refine.json', 'r'))

for key, value in add_item.items():
    if not value:
        continue
    for v_key, v_value in value.items():
        if not raw[key]:
            dic = {v_key: v_value}
            raw[key] = dic
        else:
            dic = {v_key: v_value}
            print(type(raw[key]))
            print(raw[key])
            raw[key].update(dic)

json.dump(raw, open(f'complementary_gemini_graph_all_refine.json', 'w'), indent=4)
