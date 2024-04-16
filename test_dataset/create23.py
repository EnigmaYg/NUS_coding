import json
import csv

if __name__ == "__main__":
    data = []
    with open("/Users/milesyzm/Downloads/test_dataset/2021_events_2.csv", 'r') as f:
        csv_reader = csv.reader(f)
        # cnt = 0
        for row in csv_reader:
            '''if cnt == 2:
                break
            cnt += 1'''
            merged_string = "".join(row)
            content = merged_string.split('\t')
            # print(content)
            # data.append(row[0])
            try:
                if content[2] == " ":
                    content[2] = " "
            except:
                continue


            id = set()
            with open('test_id.json', 'r') as f:
                 id_ = json.load(f)
            for key, value in id_.items():
                id.add(key)
            di = set()
            with open('test_id_0.json', 'r') as f:
                _id_ = json.load(f)
                for key, value in _id_.items():
                    di.add(key)

            fin = id.difference(di)
            if content[2] in fin:
                di.add(content[2])
                test0 = {}
                for x in di:
                    test0[x] = "0"
                with open('test_id_0.json', 'w') as f:
                    json.dump(test0, f, indent=2)
                with open('test_md5_all.json', 'r') as f:
                    test_md5 = json.load(f)
                md5_test = {}
                for key,value in test_md5.items():
                    md5_test[key] = value
                md5_test[content[5]] = content[2]
                with open('test_md5_all.json', 'w') as f:
                    json.dump(md5_test, f, indent=2)
            # for entity
            ent = set()
            with open('entities.json', 'r') as f:
                en = json.load(f)
                for key, value in en.items():
                    ent.add(key)
            ent_ = set()
            with open('entities_0.json', 'r') as f:
                en_ = json.load(f)
                for key, value in en_.items():
                    ent_.add(key)
            fin_ = ent.difference(ent_)
            origin = {}
            with open('reversed_dict_second_cleaned.json', 'r') as f:
                origin_entity = json.load(f)
                for key, value in origin_entity.items():
                    origin[key] = value
            s_ = o_ = " "
            try:
                s_ = origin[content[1]]
            except:
                pass
            try:
                o_ = origin[content[4]]
            except:
                pass
            if s_ in fin_ or o_ in fin_:
                if s_ in fin_:
                    ent_.add(s_)
                    # print(s_)
                if o_ in fin_:
                    ent_.add(o_)
                    # print(o_)
                test1 = {}
                for x in ent_:
                    test1[x] = "0"
                with open('entities_0.json', 'w') as f:
                    json.dump(test1, f, indent=2)
                with open('test_md5_all.json', 'r') as f:
                    test_md5 = json.load(f)
                md5_test = {}
                for key,value in test_md5.items():
                    md5_test[key] = value
                md5_test[content[5]] = content[2]
                with open('test_md5_all.json', 'w') as f:
                    json.dump(md5_test, f, indent=2)

            if fin == set() and fin_ == set():
                print("This is over!")
                break