# Activity Clustering using Multimodal Deep Learning for Android Applications

## Overview
This project is a Torch implementation for our paper, which activity clustering using multimodal deep learning for android applications.

## Hardware
The models are trained using folloing hardware:
- Ubuntu 18.04.5 LTS
- NVIDA TITAN Xp 12GB
- Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz
- 32GB RAM

## Dependencies
- Python version is 3.6.7

We use the following version of Pytorch.

- torch==1.1.0
- torchtext==0.3.1
- numpy==1.16.1
- scikit-learn==0.23.1
- seaborn==0.9.0
- tqdm==4.31.1
- matplotlib==3.0.3
- pandas==0.25.0
- Etc. (Included in "requirements.txt")

## Prerequisite
- Use Tkinter
```
$ sudo apt-get install python3-tk
```

- Use virtualenv
```
$ sudo apt-get install build-essential libssl-dev libffi-dev python-dev
$ sudo apt install python3-pip
$ sudo pip3 install virtualenv
$ virtualenv -p python3 env3
$ . env3/bin/activate
$ # code your stuff
$ deactivate
```

## Datasets
### Train dataset
Our dataset is based on the dataset provided by RICO.

https://storage.cloud.google.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip

### Test & validataion datasets
[Link1](https://drive.google.com/file/d/1QawDg9evv8pCx4Eoyg-9EIrJAtjFE93c/view?usp=sharing), [Link2](https://drive.google.com/file/d/17V7VioRXlrcaLo8yRDXmAM2l1mm4MdyP/view?usp=sharing)

## HOW TO EXECUTE OUR MODEL?
## Data Processing
Generate training data based on the RICO dataset and download the RICO latent vector.
```
$ . ./data_processing.sh
```

output
```
 autoencoder/seq2seq/data/all_data.txt
 autoencoder/seq2seq/data/test_23_data.txt
 autoencoder/seq2seq/data/test_34_data.txt
 autoencoder/seq2seq/data/train_data.txt
 autoencoder/seq2seq/data/val_data.txt
 data/rico/
 data_processing/activity/
 data_processing/semantic_annotations/
 full_data/rico/
```

## Seq2seq autoencoder training and vector extraction
Train the data 5-iterations with the seq2seq autoencoder and extract the latent vector.
```
$ cd autoencoder/seq2seq
$ . ./train.sh
```

output (n: 0, ... 4)
```
 autoencoder/seq2seq/log/
 data/seq2seq_23_n/
 data/seq2seq_34_n/
 full_data/seq2seq_n/
```

## Conv autoencoder training and vector extraction
Train the data 5-iterations with the conv autoencoder and extract the latent vector.
```
$ cd autoencoder/conv
$ . ./train.sh
```
output (n: 0, ..., 4)
```
 autoencoder/conv/log/
 data/conv_se_23_n/
 data/conv_re_23_n/
 data/conv_se_34_n/
 data/conv_re_34_n/
 full_data/conv_re_n/
 full_data/conv_se_n/
```

## Test data extraction and data fusion
Test data is extracted based on pre-categorized ground truth and data fusion is performed with weight.
```
$ cd ../../data
$ . ./fusion.sh
```

output (n: 0, ..., 4, f: add, cat,  m: 1, ..., 9)
```
 /data/conv_re_23_n_conv_se_23_n_f_0.m
 /data/rico_23_conv_re_23_n_f_0.m
 /data/rico_23_conv_se_23_n_f_0.m
 /data/rico_23_seq2seq_23_n_f_0.m
 /data/seq2seq_23_n_conv_re_23_n_f_0.m
 /data/seq2seq_23_n_conv_se_23_n_f_0.m
 /data/conv_re_34_n_conv_se_34_n_f_0.m
 /data/rico_34_conv_re_34_n_f_0.m
 /data/rico_34_conv_se_34_n_f_0.m
 /data/rico_34_seq2seq_34_n_f_0.m
 /data/seq2seq_34_n_conv_re_34_n_f_0.m
 /data/seq2seq_34_n_conv_se_34_n_f_0.m
```

## Clustering
Perform data clustering.
```
$ cd ../clustering
$ . ./clustering.sh
```

output
```
 clustering/result/
 clustering/visualization/
```

## Evaluation
The clustering result is evaluated by Purity, Normalized Mutual Information (NMI), and Adjusted Rand index (ARI).
```
$ cd ../evaluation
$ . ./evaluation.sh
```

output
```
 evaluation/csv/
```

## Nearest neighbor search
You can compare the results of the nearest neighbor search for *all* Rico dataset.
Fuse again for all data to compare the results of best multimodals.
```
$ cd ../full_data
$ . ./fusion.sh
```

output
```
 full_data/conv_re_0_n_conv_se_0_cat_0.3
 full_data/rico_seq2seq_1_cat_0.7
```
And execute to search for all dataset.
Finally the top 6 images of the search results are saved.
```
$ cd ../search
$ . ./search.sh
```

output
```
search/result/
```
## Our experimental results
[Link1](https://drive.google.com/file/d/1CHSsMy0Uh7UrVK3Wfbj2aeFKcSOAkkU_/view?usp=sharing), [Link2](https://drive.google.com/file/d/1X_S3XEr5NhJfQOB2GxKfp0Yf_SAxNeJV/view?usp=sharing)

## For torchtext >= 0.9.1
Please, change **torch** to **torch.legacy** in autoencoder/seq2seq python files
