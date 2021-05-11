import json
import os
from tqdm import tqdm
import glob

train_data = open("autoencoder/seq2seq/data/train_data.txt", 'w')
val_data = open("autoencoder/seq2seq/data/val_data.txt", 'w')

train_data_path = []
val_data_path = []

for dir in tqdm(glob.glob("data_processing/semantic_annotations/json/train/*")):
    train_data_path.append(dir)

for dir in tqdm(glob.glob("data_processing/semantic_annotations/json/val/*")):
    val_data_path.append(dir)

partition = json.loads(open("autoencoder/seq2seq/data/partition_128.json").read())

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
for p in tqdm(train_data_path):
    json_data = open(p).read()
    data = json.loads(json_data)
    result = isComponentLabel(data)
    bounds = isBounds(data)
    partitions = boundsToPartitions(bounds[1:])
    if result:
        train_data.write("%s\t%s\t%s\t%s\n" % (result[1:], partitions, result[1:], partitions))

print("val data generate...")
for p in tqdm(val_data_path):
    json_data = open(p).read()
    data = json.loads(json_data)
    result = isComponentLabel(data)
    bounds = isBounds(data)
    partitions = boundsToPartitions(bounds[1:])
    if result:
        val_data.write("%s\t%s\t%s\t%s\n" % (result[1:], partitions, result[1:], partitions))

print("done...")
