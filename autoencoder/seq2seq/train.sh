#!/bin/bash

mkdir log
mkdir log/check_point
mkdir log/plot
mkdir log/pth

export PYTHONPATH=.
echo 'Start seq2seq autuencoder training...'
python main/train.py -i 5

echo 'vector save...'
python main/vector_save.py -i 5

echo 'full vector save...'
python main/full_vector_save.py -i 5

echo 'done...'
