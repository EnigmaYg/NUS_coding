import json

save = {}
for i in range(0,8):
    refine = json.load(open(f'summary_gemini_wo_event{i}_refine.json', 'r'))
    for key, value in refine.items():
        save[key] = value

json.dump(save, open(f'summary_all.json', 'w'), indent=4)
