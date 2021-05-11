import glob
import json
import tqdm

data = {'34':{}, '23':{}}

for dir in tqdm.tqdm(glob.glob('activity_cluster_category_34/test_image/*')):
    data['34'][dir.split('/')[-1]] = list()
    for _dir in glob.glob(dir+'/*'):
        data['34'][dir.split('/')[-1]].append(_dir.split('/')[-1].split('.')[0])

for dir in tqdm.tqdm(glob.glob('activity_cluster_category_23/test_image/*')):
    data['23'][dir.split('/')[-1]] = list()
    for _dir in glob.glob(dir+'/*'):
        data['23'][dir.split('/')[-1]].append(_dir.split('/')[-1].split('.')[0])

with open('ground_truth_list.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
