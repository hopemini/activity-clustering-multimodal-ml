#!/bin/bash
mkdir result

export PYTHONPATH=.
python nearest_neighbor_search.py -d rico
python nearest_neighbor_search.py -d seq2seq_0
python nearest_neighbor_search.py -d conv_re_0
python nearest_neighbor_search.py -d conv_se_0
python nearest_neighbor_search.py -d conv_re_0_conv_se_0_cat_0.3
python nearest_neighbor_search.py -d rico_seq2seq_1_cat_0.7
