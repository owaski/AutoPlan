import json
from tqdm import tqdm

from envs.wikienv import WikiEnv

n = 300

print('Reading dev data...')
json_path = 'path_to_hotpotqa/hotpot_dev_fullwiki_v1.json'
with open(json_path, 'r') as r:
    dev_full_data = json.load(r)

print('Reading training data...')
json_path = 'path_to_hotpotqa/hotpot_train_v1.1.json'
with open(json_path, 'r') as r:
    train_full_data = json.load(r)

wiki = WikiEnv()

def filter(data):
    filtered_data = []
    for idx in tqdm(range(n)):
        mark = True
        for c in data[idx]['context']:
            keyword = c[0]
            try:
                wiki.search_step(keyword)
                if 'Available entities in the database' in wiki.obs:
                    mark = False
            except:
                mark = False
        if mark:
            filtered_data.append(data[idx])
    return filtered_data

print('Filtering dev data...')
filtered_dev_full_data = filter(dev_full_data)
print('Filtering training data...')
filtered_train_full_data = filter(train_full_data)

with open('path_to_hotpotqa/hotpot_dev_v1_filtered.json', 'w') as w:
    json.dump(filtered_dev_full_data, w)

with open('path_to_hotpotqa/hotpot_train_v1.1_filtered.json', 'w') as w:
    json.dump(filtered_train_full_data, w)


