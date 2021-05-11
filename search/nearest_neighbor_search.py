from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import json
import argparse
from tqdm import tqdm
import os
import json

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='rico',
                 type=str, help="data name")
par.add_argument("-c", "--class_num", default='34',
                 type=str, help="number of class")
args = par.parse_args()

data_add_names = json.loads(open("../full_data/" + args.data_name +
                 "/" + args.data_name + '_names.json').read())
data_add_data = np.load("../full_data/" + args.data_name +
                "/" + args.data_name + '_data.npy')

activity_path = "../data_processing/activity/image/all/1/"

result = data_add_data
result_data = dict()

with open('category_list.json', 'r') as f:
    category_list = json.load(f)

result_path = "result/" + args.data_name
if not os.path.exists(result_path):
    os.mkdir(result_path)

for cl, clb in category_list[args.class_num].items():
    res_path = result_path+"/"+cl
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    inp = data_add_names['name'].index(clb)
    for i in tqdm(range(len(result))):
        result_data[str(i)] = euclidean_distances(
                result[inp:inp+1], result[i:i+1]).tolist()[0][0]

    r = sorted(result_data.items(), key=lambda x: x[1])

    for i in range(0, 7, 1):
        if data_add_names['name'][int(r[i][0])] == clb:
            continue
        os.system("cp " + activity_path +
                  data_add_names['name'][int(r[i][0])] + ".jpg " + res_path)
        print("TOP%d - Image name: %s, Distance: %0.3f" % (i, data_add_names['name'][int(r[i][0])], r[i][1]))
    print("\n")
