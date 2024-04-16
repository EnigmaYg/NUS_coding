import json

refine1 = json.load(open('classification_gemini_graph_all_new.json', 'r'))
refine2 = json.load(open('classification_gemini_all.json', 'r'))
save = {}
cnt = 0

for key, value in refine2.items():
    for v_key, v_value in value.items():
        if v_value == 'aligned':
            try:
                if refine1[key][v_key] == 'aligned':
                    save[key] = v_key
                    cnt += 1
                    break
            except:
                continue
    if cnt >= 100:
        break

json.dump(save, open(f'text_alin.json', 'w'), indent=4)

'''cnt = 0
for key, value in refine.items():
    for v_key, v_value in value.items():
        if v_value == 'aligned':
            save[key] = v_key
            cnt += 1
            break
    if cnt >= 100:
        break

json.dump(save, open(f'graph_alin.json', 'w'), indent=4)'''