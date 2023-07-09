# An empirical study on Activity Clustering of Android Applications

## Overview
This project is a Torch implementation for our paper, which activity clustering using multimodal deep learning of android applications.

This repository contains source code for paper [An Empirical Study on Multimodal Activity Clustering of Android Applications](https://ieeexplore.ieee.org/abstract/document/10138412)
```
@ARTICLE{choi2023twostage,
  author={Choi, Sungmin and Seo, Hyeon-Tae and Han, Yo-Sub},
  journal={IEEE Access}, 
  title={An Empirical Study on Multimodal Activity Clustering of Android Applications}, 
  year={2023},
  volume={11},
  number={},
  pages={53598-53614},
  doi={10.1109/ACCESS.2023.3280985}
}
```

## Hardware
The models are trained using following hardware:
- Ubuntu 18.04.5 LTS
- NVIDA TITAN Xp 12GB
- Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz
- 32GB RAM

## Dependencies
- Python version is 3.6.7

We use the following versions of Pytorch.

- torch==1.1.0
- torchtext==0.3.1
- numpy==1.16.1
- scikit-learn==0.23.1
- seaborn==0.9.0
- tqdm==4.31.1
- matplotlib==3.0.3
- pandas==0.25.0
- Etc. (Included in "requirements.txt")

We also checked the following versions of Pytorch

- matplotlib==3.3.4
- numpy==1.19.5
- pandas==1.1.5
- Pillow==8.4.0
- scikit-learn==0.24.2
- scipy==1.5.4
- seaborn==0.11.2
- six==1.15.0
- torch==1.8.1+cu111
- torchaudio==0.8.1
- torchtext==0.9.1
- torchvision==0.9.1+cu111
- tqdm==4.62.3

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

- Use git lfs for large dataset file
```
$ sudo apt install git-lfs
```

## Datasets
### Rico dataset
- Our dataset is based on the dataset provided by RICO.
    - https://storage.cloud.google.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip

- **Check dataset size**
```
$ ls data_processing/rico_dataset_v0.1_semantic_annotations.zip 
154108 data_processing/rico_dataset_v0.1_semantic_annotations.zip
```
- If dataset size does not match above size (around 150MB), please download [Rico dataset](https://storage.cloud.google.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip) manually, and move to *data_processing* directory.

### Test & validataion datasets
[Description](https://github.com/hopemini/activity-clustering-multimodal-ml/tree/main/ground_truth)
[C23](https://github.com/hopemini/activity-clustering-multimodal-ml/tree/main/ground_truth/activity_cluster_category_23)
[R34](https://github.com/hopemini/activity-clustering-multimodal-ml/tree/main/ground_truth/activity_cluster_category_34)

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
Train the data 30 iterations with the seq2seq autoencoder and extract the latent vector.
```
$ cd autoencoder/seq2seq
$ . ./train.sh
```

output (n: 0, ..., 29)
```
 autoencoder/seq2seq/log/
 data/seq2seq_23_n/
 data/seq2seq_34_n/
 full_data/seq2seq_n/
```

## Conv autoencoder training and vector extraction
Train the data 30 iterations with the conv autoencoder and extract the latent vector.
```
$ cd autoencoder/conv
$ . ./train.sh
```
output (n: 0, ..., 29)
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

output (n: 0, ..., 29, f: add, cat,  m: 1, ..., 9)
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
30 iterations


# C23
| | Fusion | ARI | Fusion | NMI | Fusion | Purity |
|---|---|---|---|---|---|---|
|1|  R &#8917; S ($\mathcal{G}$ 0.7)  | 0.357 |  S + se ($\mathcal{G}$ 0.9) | 0.604 |  R &#8917; S ($\mathcal{K}$ 0.7) | 0.433 |
|2 |  R &#8917; S ($\mathcal{K}$ 0.6) | 0.346 | S + se ($\mathcal{K}$ 0.7) | 0.603 |  R + S ($\mathcal{K}$ 0.7) | 0.431 |
|3  |  R &#8917; S ($\mathcal{K}$ 0.7) | 0.345 | S + se ($\mathcal{K}$ 0.9) | 0.603 |  re + se ($\mathcal{G}$ 0.1) | 0.429  |
|4  |  R + S ($\mathcal{G}$ 0.7)  | 0.344 | S &#8917; se ($\mathcal{G}$ 0.9) | 0.602 | R &#8917; S ($\mathcal{K}$ 0.6) | 0.424 |
|5  | R + S ($\mathcal{K}$ 0.7) | 0.342 | R + se ($\mathcal{G}$ 0.8) | 0.602 | S &#8917; se ($\mathcal{G}$ 0.8) | 0.420 |
|* | R ($\mathcal{G}$) | 0.259 |  R ($\mathcal{G}$) | 0.543 |  R ($\mathcal{G}$) | 0.346 |
|* | S ($\mathcal{K}$) | 0.290  |  S ($\mathcal{K}$) | 0.546 |  S ($\mathcal{G}$) | 0.371 |
|* | re ($\mathcal{K}$) | 0.168 |  re ($\mathcal{K}$) | 0.463 | re ($\mathcal{K}$) | 0.251 |
|* | se ($\mathcal{K}$) | 0.329 |  se ($\mathcal{K}$) | 0.599 |  se ($\mathcal{K}$) | 0.400 |

# R34
| | Fusion | ARI | Fusion | NMI | Fusion | Purity |
|---|---|---|---|---|---|---|
| 1 | R &#8917; S ($\mathcal{K}$ 0.7) | 0.396 | S + se ($\mathcal{G}$ 0.9) | 0.642 | R &#8917; S ($\mathcal{K}$ 0.7) | 0.513 |
| 2 | R &#8917; S ($\mathcal{K}$ 0.6)  | 0.380 | S &#8917; se ($\mathcal{G}$ 0.9) | 0.642 | R &#8917; S ($\mathcal{K}$ 0.8) | 0.501 |
| 3 | R + S ($\mathcal{K}$ 0.7)  | 0.376 | re &#8917; se ($\mathcal{K}$ 0.1) | 0.642 | R + S ($\mathcal{K}$ 0.7) | 0.498 |
| 4 | R &#8917; S ($\mathcal{G}$ 0.7)  | 0.374 | R &#8917; S ($\mathcal{K}$ 0.7) | 0.642 | R + S ($\mathcal{G}$ 0.7) | 0.494 |
| 5 | R + S ($\mathcal{G}$ 0.7)  | 0.373 | S &#8917; se ($\mathcal{K}$ 0.9) | 0.641 | R &#8917; S ($\mathcal{G}$ 0.7) | 0.492 |
| * |  R ($\mathcal{K}$) | 0.278 |  R ($\mathcal{K}$) | 0.587 | R ($\mathcal{G}$) | 0.434 |
| * |  S ($\mathcal{K}$) | 0.313 | S ($\mathcal{K}$) | 0.578 | S ($\mathcal{K}$) | 0.420 |
| * | re ($\mathcal{K}$) | 0.213 | re ($\mathcal{K}$) | 0.525 | re ($\mathcal{K}$) | 0.309 |
| * | se ($\mathcal{K}$) | 0.344 | se ($\mathcal{K}$) | 0.636 |  se ($\mathcal{K}$) | 0.439 |

Top five  ARI, NMI, and  Purity scores for the C23 and R34. * is the result of single modals. 
The Fusion column is expressed in the form "fusion_method (clustering_algorithm  weight)".
$\mathcal{G}$ and $\mathcal{K}$ stand for GMM and K-Means.
R, S, re, and se represent Rico, seq2seq, real activity and semantic activity, respectively.
&#8917; and + mean concatenation and sum functions

## For torchtext >= 0.9.1
Please, change all **torchtext** to **torchtext.legacy** in autoencoder/seq2seq python files
```
$ git am patch/0001-Change-torchtext-to-torchtext.legacy-for-torchtext-0.patch
```
