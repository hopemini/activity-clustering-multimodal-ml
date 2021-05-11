#!/bin/bash

echo 'Downloading Rico semantic dataset...'
wget https://storage.cloud.google.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip -P data_processing
cd data_processing
unzip semantic_annotations.zip
cd semantic_annotations
mkdir image
mkdir image/all
mkdir image/all/1
mkdir json
mkdir json/all
mv *.png image/all/1
mv *.json json/all
cd ../../

echo 'Downloading Rico activity dataset...'
wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz -P data_processing

cd data_processing
tar -xvf unique_uis.tar.gz
mv combined/ activity
rm unique_uis.tar.gz
cd activity
rm *json
mkdir image
mkdir image/all
mkdir image/all/1
mv *.jpg image/all/1
cd ../../

export PYTHONPATH=.
pip install -r requirements.txt
echo 'Data split'
python data_processing/data_split.py

echo "Copying is in progress. It looks like it's stuck, but it's just a time-consuming task."
cp -rf data_processing/activity/image/all/1 data_processing/activity/image/all_dir/
cp data_processing/dir_classifications.py data_processing/activity/image/all_dir/
cp -rf data_processing/semantic_annotations/image/all/1 data_processing/semantic_annotations/image/all_dir
cp data_processing/dir_classifications.py data_processing/semantic_annotations/image/all_dir/

echo 'Data dir classification'
cd data_processing/activity/image/all_dir
export PYTHONPATH=.
python dir_classifications.py -t jpg
cd ../../../semantic_annotations/image/all_dir
export PYTHONPATH=.
python dir_classifications.py -t png
cd ../../../../
export PYTHONPATH=.

echo 'Downloading Rico latent vectors...'
mkdir data/rico
wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_layout_vectors.zip -P data/rico
cd data/rico
unzip ui_layout_vectors.zip
cd ui_layout_vectors
mv * ../
cd ../
rm -rf ui_layout_vectors ui_layout_vectors.zip
mv ui_names.json rico_names.json
mv ui_vectors.npy rico_data.npy
sed -i 's/ui_names/name/g' rico_names.json
sed -i 's/.png//g' rico_names.json
cd ../../
cp -rf data/rico/ full_data/

echo 'Seq2seq Autoencoder data generate'
echo 'Training data generate'
python autoencoder/seq2seq/data/training_data_generator.py
echo 'All data generate'
python autoencoder/seq2seq/data/all_data_generator.py
echo 'Test data generate'
python autoencoder/seq2seq/data/test_data_generator.py

echo 'Done...'
