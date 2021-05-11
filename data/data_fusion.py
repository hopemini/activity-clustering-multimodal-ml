from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer

import numpy as np
import json
import argparse
import os
from tqdm import tqdm

MAX_SIZE = 72200

par = argparse.ArgumentParser()
par.add_argument("-d1", "--data1", default='rico',
                 type=str, help="Select the first data to merge")
par.add_argument("-d2", "--data2", default='seq2seq',
                 type=str, help="Select the second data to merge")
par.add_argument("-t", "--type", default='cat',
                 type=str, help="merge type(add/cat/evg)")
par.add_argument("-s", "--scaler", default=None,
                 type=str, help="Select the scaler(minmax/maxabs)")
par.add_argument("-w", "--weight", default=None,
                 type=float, help="Set weight")
args = par.parse_args()

d2_names = json.loads(open(args.data2 + "/" + 
        args.data2 + '_names.json').read())
d2_data = np.load(args.data2 + "/" + 
        args.data2 + '_data.npy')

d1_names = json.loads(open(args.data1 + "/" + 
        args.data1 + '_names.json').read())
d1_data = np.load(args.data1 + "/" + 
        args.data1 + '_data.npy')

#print("{}_data shape = {}".format(args.data1, d1_data.shape))
#print("{}_data shape = {}".format(args.data2, d2_data.shape))

if args.scaler == 'minmax':
    print("scaler : {}".format(args.scaler))
    d1_data = MinMaxScaler().fit_transform(d1_data)
    d2_data = MinMaxScaler().fit_transform(d2_data)
elif args.scaler == 'maxabs':
    print("scaler : {}".format(args.scaler))
    d1_data = MaxAbsScaler().fit_transform(d1_data)
    d2_data = MaxAbsScaler().fit_transform(d2_data)
elif args.scaler == 'robust':
    print("scaler : {}".format(args.scaler))
    d1_data =RobustScaler(quantile_range=(25, 75)).fit_transform(d1_data)
    d2_data =RobustScaler(quantile_range=(25, 75)).fit_transform(d2_data)
elif args.scaler == 'QT-norm':
    print("scaler : {}".format(args.scaler))
    d1_data = QuantileTransformer(output_distribution='normal').fit_transform(d1_data)
    d2_data = QuantileTransformer(output_distribution='normal').fit_transform(d2_data)
elif args.scaler == 'QT-uni':
    print("scaler : {}".format(args.scaler))
    d1_data = QuantileTransformer(output_distribution='uniform').fit_transform(d1_data)
    d2_data = QuantileTransformer(output_distribution='uniform').fit_transform(d2_data)
elif args.scaler == 'standard':
    print("scaler : {}".format(args.scaler))
    d1_data = StandardScaler().fit_transform(d1_data)
    d2_data = StandardScaler().fit_transform(d2_data)
elif args.scaler == 'PT-yj':
    print("scaler : {}".format(args.scaler))
    d1_data = PowerTransformer(method='yeo-johnson').fit_transform(d1_data)
    d2_data = PowerTransformer(method='yeo-johnson').fit_transform(d2_data)
elif args.scaler == 'normalizer':
    print("scaler : {}".format(args.scaler))
    d1_data = Normalizer().fit_transform(d1_data)
    d2_data = Normalizer().fit_transform(d2_data)
else:
    print("scaler : {}".format(args.scaler))

print("fusion type : {}".format(args.type))

if args.weight is not None:
    weight = "_" + str(args.weight)
else:
    weight = ""

if args.scaler is not None:
    _scaler = "_" + args.scaler
else:
    _scaler = ""

data_path = args.data1 + "_" + args.data2 + "_" + args.type + _scaler + weight

print("data name : {}".format(data_path))

cnt = 0
for i in tqdm(range(MAX_SIZE)):
    if str(i) in d2_names['name'] and str(i) in d1_names['name']:
        cnt += 1
        temp_names = str(i)
        if args.weight is None:
            if args.type == 'add':
                temp = np.array([(d2_data[d2_names['name'].index(str(i))]
                        + d1_data[d1_names['name'].index(str(i))])])
            elif args.type == 'cat':
                temp = np.array([np.concatenate((d2_data[d2_names['name'].index(str(i))],
                        d1_data[d1_names['name'].index(str(i))]), axis=0)])
            elif args.type == 'evg':
                temp = np.array([(d2_data[d2_names['name'].index(str(i))]
                        + d1_data[d1_names['name'].index(str(i))])])
        else:
            if args.type == 'add':
                temp = np.array([(d2_data[d2_names['name'].index(str(i))]*(1-args.weight)
                        + d1_data[d1_names['name'].index(str(i))]*args.weight)])
            elif args.type == 'cat':
                temp = np.array([np.concatenate((d2_data[d2_names['name'].index(str(i))]*(1-args.weight),
                        d1_data[d1_names['name'].index(str(i))]*args.weight), axis=0)])
            elif args.type == 'evg':
                temp = np.array([(d2_data[d2_names['name'].index(str(i))]*(1-args.weight)
                        + d1_data[d1_names['name'].index(str(i))]*args.weight)])

        if cnt == 1:
            result_names = [temp_names]
            result = temp
        else:
            result_names.append(temp_names)
            result = np.concatenate((result, temp), axis=0)

if not os.path.exists(data_path):
    os.mkdir(data_path)

name_dic = {}

np.save(data_path + "/" + data_path + '_data.npy', result)
name_dic["name"] = result_names
name_json = json.dumps(name_dic)
name_file = open(data_path + "/" + data_path + "_names.json","w")
name_file.write(name_json)
name_file.close()

print("done.")
