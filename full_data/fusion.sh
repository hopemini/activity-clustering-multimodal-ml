#!/bin/bash

export PYTHONPATH=.

echo 'Data fusion...'

python data_fusion.py -d1 conv_re_0 -d2 conv_se_0 -t cat -w 0.3
python data_fusion.py -d1 rico -d2 seq2seq_1 -t cat -w 0.7

echo 'Done...'
