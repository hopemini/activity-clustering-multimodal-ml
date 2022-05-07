#!/bin/bash

mkdir log
mkdir log/check_point
mkdir log/plot
mkdir log/pth

export PYTHONPATH=.
echo 'create training data...'
python data/training_data_generator_random.py -n 15000 -i 5

echo 'Start seq2seq autuencoder training...'
python main/train_random.py -n 15000 -i 5

echo 'vector save...'
python main/vector_save_random.py -n 15000 -i 5

#echo 'full vector save...'
#python main/full_vector_save.py -i 5

echo 'done...'
