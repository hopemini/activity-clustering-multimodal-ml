import json
import os
from tqdm import tqdm
import glob
import random
import argparse

par = argparse.ArgumentParser()
par.add_argument("-i", "--iteration", default=5,
                 type=int, help="number of iteration")
par.add_argument("-n", "--number_of_train", default=64864,
                 type=int, help="number of train")
args = par.parse_args()
num = args.number_of_train

#val_data = open("data/val_data.txt", 'w')

train_data_path = []
val_data_path = []

for dir in tqdm(glob.glob("../../data_processing/semantic_annotations/json/train/*")):
    train_data_path.append(dir)

#for dir in tqdm(glob.glob("../../data_processing/semantic_annotations/json/val/*")):
#    val_data_path.append(dir)

partition = json.loads(open("data/partition_128.json").read())

def isComponentLabel(data):
    result = ""
    if data.get("componentLabel"):
        result = result + " " + data["componentLabel"].replace(" ","")
    if data.get("children"):
        for i in data["children"]:
            result = result + isComponentLabel(i)
    return result

def isBounds(data):
    result = [[]]
    if data.get("componentLabel"):
        for i in data["bounds"]:
            result[0].append(i)
    if data.get("children"):
        for i in data["children"]:
            result.extend(isBounds(i))
    return result

def boundsToPartitions(bounds):
    pars = []
    for b in bounds:
        c = ""
        for _, p in partition.items():
            if(p[0] < b[2] and p[2] > b[0] and p[1] < b[3] and p[3] > b[1]):
                c += "1"
            else:
                c += "0"

        a = ""
        for i in range(0,128,4):
            a += format(int(c[i:i+4],2), 'x')

        pars.append(a)

    return " ".join(pars)

print("train data generate...")
se_img_path = "../../data_processing/semantic_annotations/image/"
re_img_path = "../../data_processing/activity/image/"
for _iter in range(args.iteration):
    train_data_path_random = []
    train_data_path_random = random.sample(train_data_path, num)
    train_path = str(num) + '_' + str(_iter) + '/'
    if not os.path.exists(se_img_path + train_path):
        os.mkdir(se_img_path + train_path)
    if not os.path.exists(re_img_path + train_path):
        os.mkdir(re_img_path + train_path)

    train_path = str(num) + '_' + str(_iter) + '/1/'
    if not os.path.exists(se_img_path + train_path):
        os.mkdir(se_img_path + train_path)
    if not os.path.exists(re_img_path + train_path):
        os.mkdir(re_img_path + train_path)

    for files in tqdm(train_data_path_random):
        f = files.split('/')[-1].split('.')[0]
        os.system('ln -s ../../all/1/' + f + '.png ' + se_img_path + train_path)
        os.system('ln -s ../../all/1/' + f + '.jpg ' + re_img_path + train_path)

    train_data = open("data/train_data_" + str(num) + "_" + str(_iter) +  ".txt", 'w')
    for p in tqdm(train_data_path_random):
        json_data = open(p).read()
        data = json.loads(json_data)
        result = isComponentLabel(data)
        bounds = isBounds(data)
        partitions = boundsToPartitions(bounds[1:])
        if result:
            train_data.write("%s\t%s\t%s\t%s\n" % (result[1:], partitions, result[1:], partitions))

#print("val data generate...")
#for p in tqdm(val_data_path):
#    json_data = open(p).read()
#    data = json.loads(json_data)
#    result = isComponentLabel(data)
#    bounds = isBounds(data)
#    partitions = boundsToPartitions(bounds[1:])
#    if result:
#        val_data.write("%s\t%s\t%s\t%s\n" % (result[1:], partitions, result[1:], partitions))

print("done...")
