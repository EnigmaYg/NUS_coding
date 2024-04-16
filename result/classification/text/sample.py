import json

refine = json.load(open('classification_gemini_all.json', 'r'))

save = {}
cnt = 0
for key, value in refine.items():
    for v_key, v_value in value.items():
        if v_value == 'complementary':
            save[key] = v_key
            cnt += 1
            break
    if cnt >= 100:
        break

json.dump(save, open(f'text_comp.json', 'w'), indent=4)