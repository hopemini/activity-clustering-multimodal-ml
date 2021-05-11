import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default="rico_seq2seq",
                 type=str, help="data name")
par.add_argument("-t", "--fusion_type", default=None, choices=["add", "cat"],
                 type=str, help="fusion type(add/cat)")
par.add_argument("-s", "--scaler", default=None,
                 type=str, help="Select the scaler")
par.add_argument("-w", "--weight", default=None,
                 type=str, help="rico weight")
par.add_argument("-i", "--iteration", default=1,
                 type=int, help="number of iteration")
par.add_argument("-k", "--number_of_classes", default=18,
                 type=int, help="number of classes")
args = par.parse_args()

activity_path = "../data_processing/activity/image/all/1/"

def visualization(data, predict, data_name, vis_path, k):
    predict = pd.DataFrame(predict)
    predict.columns=['predict']
    final_df = pd.DataFrame(np.hstack((predict, data)))
    cols = list(data[0,:])
    cols.insert(0,'group')
    final_df.columns = cols
    feature_df = pd.DataFrame(data)

    # tSNE
    tsne = TSNE(n_components=2).fit_transform(feature_df)
    fig_t = plt.figure(figsize=(10,7))
    ax_t = fig_t.add_subplot(111)
    ax_t.set_xlabel('PC', fontsize = 25)
    ax_t.set_ylabel('PC', fontsize = 25)
    ax_t.set_title(data_name + "_t-SNE", fontsize = 25)
    ax_t.scatter(tsne[:,0], tsne[:,1], c=final_df['group'])
    ax_t.grid()
    fig_t.savefig(vis_path + data_name + "_k" + str(k) + "_tSNE.png")

    # PCA
    dim = 3
    pca = PCA(n_components=dim).fit_transform(feature_df)
    fig_p = plt.figure(figsize=(10,7))
    ax_p = fig_p.add_subplot(111)
    ax_p.set_xlabel('PC', fontsize = 25)
    ax_p.set_ylabel('PC', fontsize = 25)
    ax_p.set_title(data_name + "_PCA", fontsize = 25)
    ax_p.scatter(pca[:,0], pca[:,1], c=final_df['group'])
    ax_p.grid()
    fig_p.savefig(vis_path + data_name + "_k" + str(k) + "_PCA.png")

def image_classification(name_list, predict, result_path, k):
    k_result = dict()
    for i in range(len(predict)):
        if str(predict[i]) not in k_result.keys():
            k_result[str(predict[i])] = []
        k_result[str(predict[i])].append(name_list[i])

    for i in range(1,k+1):
        if not os.path.exists(result_path + str(i)):
            os.mkdir(result_path + str(i))

    for k in k_result.keys():
        for i in k_result[k]:
            os.system("cp " + activity_path +
                    i + ".jpg " + result_path + str(int(k)+1))

def save_result(name_list, predict, result_path, k):
    k_result = dict()
    for i in range(len(predict)):
        if str(predict[i]) not in k_result.keys():
            k_result[str(predict[i])] = []
        k_result[str(predict[i])].append(name_list[i])

    for k in k_result.keys():
        with open(result_path+str(int(k)+1), 'w') as f:
            for i in k_result[k]:
                f.write(i+"\n")

def kmeans(data, name_list, data_name, result_path, vis_path):
    print("Start Kmeans clustering..")
    k = args.number_of_classes

    model = KMeans(n_clusters=k, algorithm='auto', random_state=42)
    model.fit(data)
    predict = model.predict(data)

    result_path = result_path + "kmeans/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    vis_path = vis_path + "kmeans/"
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    #image_classification(name_list, predict, result_path, k)
    save_result(name_list, predict, result_path, k)
    visualization(data, predict, data_name, vis_path, k)
    print("Done.\n")

def gaussian_mixture(data, name_list, data_name, result_path, vis_path):
    print("Start Gaussian Mixture clustering..")
    k = args.number_of_classes

    model = GaussianMixture(n_components=k, init_params='kmeans', random_state=0)
    model.fit(data)
    predict = model.predict(data)

    result_path = result_path + "gaussian_mixture/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    vis_path = vis_path + "gaussian_mixture/"
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    #image_classification(name_list, predict, result_path, k)
    save_result(name_list, predict, result_path, k)
    visualization(data, predict, data_name, vis_path, k)
    print("Done.\n")

def dbscan(data, name_list, data_name, result_path, vis_path):
    print("Start DBSCAN clustering..")

    model = DBSCAN(eps=0.5, min_samples=10)
    model.fit(data)
    k = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    predict = model.fit_predict(data)

    result_path = result_path + "dbscan/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    vis_path = vis_path + "dbscan/"
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    #image_classification(name_list, predict, result_path, k)
    save_result(name_list, predict, result_path, k)
    visualization(data, predict, data_name, vis_path, k)
    print("Done.\n")

def birch(data, name_list, data_name, result_path, vis_path):
    print("Start Birch clustering..")
    k = 18

    model = Birch(n_clusters=k)
    model.fit(data)
    predict = model.predict(data)

    result_path = result_path + "birch/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    vis_path = vis_path + "birch/"
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    #image_classification(name_list, predict, result_path, k)
    save_result(name_list, predict, result_path, k)
    visualization(data, predict, data_name, vis_path, k)
    print("Done.\n")

def optics(data, name_list, data_name, result_path, vis_path):
    print("Start OPTICS clustering..")

    model = OPTICS(min_samples=10)
    model.fit(data)
    k = max(model.labels_)
    predict = model.fit_predict(data)

    result_path = result_path + "optics/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    vis_path = vis_path + "optics/"
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    #image_classification(name_list, predict, result_path, k)
    save_result(name_list, predict, result_path, k)
    visualization(data, predict, data_name, vis_path, k)
    print("Done.\n")

if __name__ == "__main__":
    fusion_type = "_" + args.fusion_type if args.fusion_type is not None else ""
    scaler = "_" + args.scaler if args.scaler is not None else ""
    weight = "_" + args.weight if args.weight is not None else ""
    data_name = args.data_name + fusion_type + scaler + weight
    data_path = "../data/" + data_name + "/"

    print("Data loading..")
    data = np.load(data_path + data_name + "_data.npy")
    names = json.loads(open(data_path + data_name + '_names.json').read())

    name_list = names['name']

    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists("visualization"):
        os.mkdir("visualization")

    result_path = "result/" + data_name + "/"
    vis_path = "visualization/" + data_name + "/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    try:
        gaussian_mixture(data, name_list, data_name, result_path, vis_path)
    except Exception as e:
        print("gaussian mixture error. ", e)
    try:
        kmeans(data, name_list, data_name, result_path, vis_path)
    except Exception as e:
        print("kmeans error. ", e)
""" we do not use the following clusterings in ICSE2021
    try:
        dbscan(data, name_list, data_name, result_path, vis_path)
    except Exception as e:
        print("DBSCAN error. ", e)
    try:
        birch(data, name_list, data_name, result_path, vis_path)
    except Exception as e:
        print("birch error. ", e)
    try:
        optics(data, name_list, data_name, result_path, vis_path)
    except Exception as e:
        print("OPTICS error. ", e)
"""
