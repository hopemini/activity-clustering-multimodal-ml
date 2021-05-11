import json
import os
import glob
from tqdm import tqdm

test_23_path = "data_processing/semantic_annotations/json/test_23/"
test_34_path = "data_processing/semantic_annotations/json/test_34/"
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
        for _,p in partition.items():
            if(p[0] < b[2] and p[2] > b[0] and p[1] < b[3] and p[3] > b[1]):
                c += "1"
            else:
                c += "0"
        a = ""
        for i in range(0,128,4):
            a += format(int(c[i:i+4],2), 'x')

        pars.append(a)
    return " ".join(pars)

test_23 = list()
for dir in glob.glob(test_23_path+"*"):
    test_23.append(dir)

test_34 = list()
for dir in glob.glob(test_34_path+"*"):
    test_34.append(dir)

with open("autoencoder/seq2seq/data/test_23_data.txt", 'w') as dataset_23:
    for json_data_path in tqdm(test_23):
        json_data = open(json_data_path).read()
        data = json.loads(json_data)
        result = isComponentLabel(data)
        bounds = isBounds(data)
        partitions = boundsToPartitions(bounds[1:])
        if result:
            dataset_23.write("%s\t%s\t%s\n" % (
                json_data_path.split("/")[-1].split(".")[0], result[1:], partitions))

with open("autoencoder/seq2seq/data/test_34_data.txt", 'w') as dataset_34:
    for json_data_path in tqdm(test_34):
        json_data = open(json_data_path).read()
        data = json.loads(json_data)
        result = isComponentLabel(data)
        bounds = isBounds(data)
        partitions = boundsToPartitions(bounds[1:])
        if result:
            dataset_34.write("%s\t%s\t%s\n" % (
                json_data_path.split("/")[-1].split(".")[0], result[1:], partitions))
