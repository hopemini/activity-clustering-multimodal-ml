import numpy as np
import json
import argparse
import os
from tqdm import tqdm
import glob

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='rico',
                 type=str, help="data name")
args = par.parse_args()

data_names = json.loads(open(args.data_name + "/" +
        args.data_name + '_names.json').read())
data = np.load(args.data_name + "/" +
        args.data_name + '_data.npy')

def run(n):
    ground_path = "../ground_truth/activity_cluster_category_" + n + "/test_image/"
    test_names = list()
    for dir in tqdm(glob.glob(ground_path+"*")):
        for f in glob.glob(dir+"/*"):
            test_names.append(f.split("/")[-1].split(".")[0])

    test_names.sort()
    test_index = list()
    for t in test_names:
        if t in data_names['name']:
            test_index.append(data_names['name'].index(t))

    test_data = list()
    for i in test_index:
        test_data.append(data[i])

    test_data_np = np.array(test_data)

    if not os.path.exists(args.data_name + "_" + n):
        os.mkdir(args.data_name + "_" + n)

    name_dic = dict()
    np.save(args.data_name + "_" + n + "/" +
            args.data_name + "_" + n + "_data.npy", test_data_np)
    name_dic["name"] = test_names
    name_json = json.dumps(name_dic)
    name_file = open(args.data_name + "_" + n + "/" +
            args.data_name + "_" + n + "_names.json","w")
    name_file.write(name_json)
    name_file.close()

if __name__ == "__main__":
    test_num = ["23", "34"]
    for n in test_num:
        run(n)
