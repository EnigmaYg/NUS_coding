import json

cnt = 0
for i in range(239,245,2):

    with open('photo-url0-999.json','r') as f:
        docs_data1 = json.load(f)
        # print(type(docs_data1))
        path1 = str(i)
        path2 = 'C:/Users/Administrator/Documents/photo-url'
        path3 = '000-'
        path4 = '999.json'
        j = i + 1
        path5 = str(j)
        path = path2 + path1 + path3 + path5 + path4
        with open(path,'r') as f:
            docs_data2 = json.load(f)
            docs_data1.update(docs_data2)
            docs_data = docs_data1
    with open('photo-url0-999.json', 'w') as f:
        json.dump(docs_data, f, indent=2)
    print(cnt)
    cnt += 1
