import json
import os
from tqdm import tqdm

data_path = "data_processing/semantic_annotations/json/all/"
partition = json.loads(open("autoencoder/seq2seq/data/partition_128.json").read())
dataset = open("autoencoder/seq2seq/data/all_data.txt", 'w')

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

for i in tqdm(range(0,80000)):
    json_data_path = data_path + str(i) + ".json"
    if os.path.isfile(json_data_path):
        json_data = open(json_data_path).read()
        data = json.loads(json_data)
        result = isComponentLabel(data)
        bounds = isBounds(data)
        partitions = boundsToPartitions(bounds[1:])
        if result:
            dataset.write("%s\t%s\t%s\n" % (str(i), result[1:], partitions))
    else:
        continue
