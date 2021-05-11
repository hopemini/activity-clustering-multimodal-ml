import glob
import argparse
import pandas as pd
import copy
import os
import json
from sklearn.metrics.cluster import  adjusted_rand_score, normalized_mutual_info_score

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default="rico_seq2seq",
                 type=str, help="data name")
par.add_argument("-e", "--evaluation", default="nmi", choices=["purity", "nmi", "ari"],
                 type=str, help="Select the evaluation method(purity, nmi, ari)")
args = par.parse_args()

clustering_algorithms = ['gaussian_mixture', 'kmeans']
fusion_types = ["add", "cat"]
weights = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
iteration = ["0", "1", "2", "3", "4"]
class_num = ["23", "34"]
conv_types = ["se", "re"]
with open('ground_truth_list.json', 'r') as f:
    ground_truth = json.load(f)

total_best_result = dict()

def Purity(data_name, clustering_result, c_num, result_dict, clustering_algorithm):
    g_result = dict()
    _ground_truth = copy.deepcopy(ground_truth)
    for c_key in clustering_result.keys():
        comp = list()
        for g_key in _ground_truth[c_num].keys():
            comp.append((g_key, c_key, len(list(
                set(ground_truth[c_num][g_key]).intersection(clustering_result[c_key]))),
                len(ground_truth[c_num][g_key])))
        comp.sort(key=lambda element:element[2])
        if len(_ground_truth[c_num].keys()) == 0:
            break
        g_result[comp[-1][0]] = list()
        g_result[comp[-1][0]].append(comp[-1][1])
        g_result[comp[-1][0]].append(comp[-1][2])
        del _ground_truth[c_num][comp[-1][0]]

    result = list()
    for g_key in ground_truth[c_num].keys():
        if g_key in g_result.keys():
            result.append((g_key, g_result[g_key][0], g_result[g_key][1], len(ground_truth[c_num][g_key])))
        else:
            result.append((g_key, "", 0, len(ground_truth[c_num][g_key])))

    matched = 0
    total = 0
    if len(result_dict["List"]) == 0:
        for t in result:
            result_dict["List"].append(t[0])
        result_dict["List"].append("Total")

    for t in result:
        result_dict[clustering_algorithm].append("%.1f" % (t[2]/t[3]*100) + "%")
        matched += t[2]
        total += t[3]

    if total_best_result[data_name] < matched/total*100:
        total_best_result[data_name] = matched/total*100

    return ("%.1f" % (matched/total*100) + "%")

def NMI(data_name, ground_truth_labels, clustering_result, c_num):
    clustering_resul_labels = copy.deepcopy(ground_truth_labels)
    for g_key in ground_truth[c_num].keys():
        for g in ground_truth[c_num][g_key]:
            ground_truth_labels[ground_truth_labels.index(g)] = g_key

    for c_key in clustering_result.keys():
        for c in clustering_result[c_key]:
            clustering_resul_labels[clustering_resul_labels.index(c)] = c_key

    nmi_score = normalized_mutual_info_score(ground_truth_labels, clustering_resul_labels)
    if total_best_result[data_name] < nmi_score:
        total_best_result[data_name] = nmi_score

    return ("%.3f" % (nmi_score))

def ARI(data_name, ground_truth_labels, clustering_result, c_num):
    clustering_resul_labels = copy.deepcopy(ground_truth_labels)
    for g_key in ground_truth[c_num].keys():
        for g in ground_truth[c_num][g_key]:
            ground_truth_labels[ground_truth_labels.index(g)] = g_key

    for c_key in clustering_result.keys():
        for c in clustering_result[c_key]:
            clustering_resul_labels[clustering_resul_labels.index(c)] = c_key

    ari_score = adjusted_rand_score(ground_truth_labels, clustering_resul_labels)
    if total_best_result[data_name] < ari_score:
        total_best_result[data_name] = ari_score

    return ("%.3f" % (ari_score))

def save_csv_file(result_dict):
    result_path = "csv/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_path = "csv/" + args.evaluation + "/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if args.evaluation != "purity":
        del result_dict["List"]
    df = pd.DataFrame(result_dict)
    df.to_csv(result_path + data_name + ".csv", index=False)

def run(data_name, c_num):
    total_best_result[data_name] = 0.0
    result_dict = dict()
    result_dict["List"] = list()
    for i, clustering_algorithm in enumerate(clustering_algorithms):
        result_dict[clustering_algorithm] = list()
        result_path = '../clustering/result/' + data_name + "/" + clustering_algorithm + "/"

        clustering_result = dict()
        clustering_result_temp = dict()

        for dirs in glob.glob(result_path+"*"):
            with open(dirs, 'r') as fd:
                res = fd.read()
            clustering_result_temp[dirs.split("/")[-1]] = " ".join(res.split('\n')).split()

        clustering_result_list = list(clustering_result_temp.keys())
        clustering_result_list.sort()
        for c_key in clustering_result_list:
            clustering_result[c_key] = clustering_result_temp[c_key]

        ground_truth_labels = list()
        for v in ground_truth[c_num].values():
            ground_truth_labels.extend(v)

        if args.evaluation == "purity":
            result_dict[clustering_algorithm].append(
                    Purity(data_name, clustering_result, c_num, result_dict, clustering_algorithm))
        elif args.evaluation == "nmi":
            result_dict[clustering_algorithm].append(
                    NMI(data_name, ground_truth_labels, clustering_result, c_num))
        elif args.evaluation == "ari":
            result_dict[clustering_algorithm].append(
                    ARI(data_name, ground_truth_labels, clustering_result, c_num))

    save_csv_file(result_dict)

if __name__ == "__main__":
    for c_num in class_num:
        data_name = "rico_" + c_num
        run(data_name, c_num)
        for _iter in iteration:
            data_name = "seq2seq_" + c_num + "_" + _iter
            run(data_name, c_num)
            for c_type in conv_types:
                data_name = "conv_" + c_type + "_" + c_num + "_" + _iter
                run(data_name, c_num)
            for f_type in fusion_types:
                data_name = "rico_" + c_num + "_seq2seq_" + c_num + "_" + _iter + "_" + f_type
                run(data_name, c_num)
            for f_type in fusion_types:
                data_name = "conv_re_" + c_num + "_" + _iter + "_conv_se_" + c_num + "_" + _iter + "_" + f_type
                run(data_name, c_num)
            for c_type in conv_types:
                for f_type in fusion_types:
                    data_name = ("rico_" + c_num + "_conv_" + c_type + "_" +
                                 c_num + "_" + _iter + "_" + f_type)
                    run(data_name, c_num)
                for f_type in fusion_types:
                    data_name = ("seq2seq_" + c_num + "_" + _iter + "_conv_" + c_type + "_" +
                                 c_num + "_" + _iter + "_" + f_type)
                    run(data_name, c_num)
            for f_type in fusion_types:
                for w in weights:
                    data_name = "rico_" + c_num + "_seq2seq_" + c_num + "_" + _iter + "_" + f_type + "_" + w
                    run(data_name, c_num)
            for f_type in fusion_types:
                for w in weights:
                    data_name = "conv_re_" + c_num + "_" + _iter + "_conv_se_" + c_num + "_" + _iter + "_" + f_type + "_" + w
                    run(data_name, c_num)
            for c_type in conv_types:
                for w in weights:
                    for f_type in fusion_types:
                        data_name = ("rico_" + c_num + "_conv_" + c_type + "_" +
                                     c_num + "_" + _iter + "_" + f_type + "_" + w)
                        run(data_name, c_num)
                    for f_type in fusion_types:
                        data_name = ("seq2seq_" + c_num + "_" + _iter + "_conv_" + c_type + "_" +
                                     c_num + "_" + _iter + "_" + f_type + "_" + w)
                        run(data_name, c_num)

    order_total_best_result = sorted(total_best_result.items(), reverse=True, key=lambda item: item[1])

    fd = open("csv/" + args.evaluation  + "/total_best_result.txt", "w")
    print("Total Result Length : {}".format(len(total_best_result)))
    fd.write("Total Result Length : {}\n".format(len(total_best_result)))

    for i, items in enumerate(order_total_best_result):
        #if i > 30:
        #    break
        if args.evaluation == "purity":
            print("Top{} = {} : {}".format(i+1, items[0], "%.1f"%items[1]+"%"))
            fd.write("Top{} = {} : {}".format(i+1, items[0], "%.1f"%items[1]+"%\n"))
        else:
            print("Top{} = {} : {}".format(i+1, items[0], "%.3f"%items[1]))
            fd.write("Top{} = {} : {}".format(i+1, items[0], "%.3f"%items[1]+"\n"))
    fd.close()
