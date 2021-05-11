import glob
from tqdm import tqdm
import os
import random

temp = list()

json_path = "data_processing/semantic_annotations/json/"
se_img_path = "data_processing/semantic_annotations/image/"
re_img_path = "data_processing/activity/image/"
test_23_path = "ground_truth/activity_cluster_category_23/test_image/"
test_34_path = "ground_truth/activity_cluster_category_34/test_image/"
val_path = "ground_truth/activity_cluster_category_23/labeled_image/"

if not os.path.isdir(json_path):
    os.mkdir(json_path)
if not os.path.isdir(json_path+"test_23"):
    os.mkdir(json_path+"test_23")
if not os.path.isdir(json_path+"test_34"):
    os.mkdir(json_path+"test_34")
if not os.path.isdir(json_path+"train"):
    os.mkdir(json_path+"train")
if not os.path.isdir(json_path+"val"):
    os.mkdir(json_path+"val")

if not os.path.isdir(se_img_path+"test_23"):
    os.mkdir(se_img_path+"test_23")
if not os.path.isdir(se_img_path+"test_34"):
    os.mkdir(se_img_path+"test_34")
if not os.path.isdir(re_img_path+"test_23"):
    os.mkdir(re_img_path+"test_23")
if not os.path.isdir(re_img_path+"test_34"):
    os.mkdir(re_img_path+"test_34")

test_23 = list()
se_test_23 = list()
re_test_23 = list()
for dir in tqdm(glob.glob(test_23_path+"*")):
    for f in glob.glob(dir+"/*"):
        test_23.append(f.split("/")[-1].split(".")[0]+".json")
        se_test_23.append(f.split("/")[-1].split(".")[0]+".png")
        re_test_23.append(f.split("/")[-1].split(".")[0]+".jpg")

test_34 = list()
se_test_34 = list()
re_test_34 = list()
for dir in tqdm(glob.glob(test_34_path+"*")):
    for f in glob.glob(dir+"/*"):
        test_34.append(f.split("/")[-1].split(".")[0]+".json")
        se_test_34.append(f.split("/")[-1].split(".")[0]+".png")
        re_test_34.append(f.split("/")[-1].split(".")[0]+".jpg")

val = list()
for dir in tqdm(glob.glob(val_path+"*")):
    for f in glob.glob(dir+"/*"):
        val.append(f.split("/")[-1].split(".")[0]+".json")

json_all = list()
for dir in tqdm(glob.glob(json_path+"all/*")):
    json_all.append(dir.split("/")[-1])

se_img_all = list()
for dir in tqdm(glob.glob(se_img_path+"all/1/*")):
    se_img_all.append(dir.split("/")[-1])

re_img_all = list()
for dir in tqdm(glob.glob(re_img_path+"all/1/*")):
    re_img_all.append(dir.split("/")[-1])

print("json data split...")
for t in tqdm(json_all):
    if t in test_34:
        os.system("cp " + json_path+"all/"+t + " " + json_path+"test_34/")
    if t in test_23:
        os.system("cp " + json_path+"all/"+t + " " + json_path+"test_23/")
    elif t in val:
        os.system("cp " + json_path+"all/"+t + " " + json_path+"val/")
    else:
        os.system("cp " + json_path+"all/"+t + " " + json_path+"train/")

print("semantic image data split...")
for t in tqdm(se_img_all):
    if t in se_test_23:
        os.system("mkdir " + se_img_path + "test_23/" + t.split(".")[0])
        os.system("cp " + se_img_path+"all/1/"+t + " " + se_img_path+"test_23/"+t.split(".")[0]+"/")
    if t in se_test_34:
        os.system("mkdir " + se_img_path + "test_34/" + t.split(".")[0])
        os.system("cp " + se_img_path+"all/1/"+t + " " + se_img_path+"test_34/"+t.split(".")[0]+"/")

print("real image data split...")
for t in tqdm(re_img_all):
    if t in re_test_23:
        os.system("mkdir " + re_img_path + "test_23/" + t.split(".")[0])
        os.system("cp " + re_img_path+"all/1/"+t + " " + re_img_path+"test_23/"+t.split(".")[0]+"/")
    if t in re_test_34:
        os.system("mkdir " + re_img_path + "test_34/" + t.split(".")[0])
        os.system("cp " + re_img_path+"all/1/"+t + " " + re_img_path+"test_34/"+t.split(".")[0]+"/")

print("done...")
