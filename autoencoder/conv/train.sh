## Traing AutoEncoder

# Image path
IMAGE_PATH='../../data_processing/'

echo 'conv autoencoder training..'
python train.py -d ${IMAGE_PATH} -t real -i 5
python train.py -d ${IMAGE_PATH} -t semantic_annotations -i 5

echo 'vector save..'
python vector_save.py -d ${IMAGE_PATH} -t real -i 5
python vector_save.py -d ${IMAGE_PATH} -t semantic_annotations -i 5

echo 'full vector save..'
python full_vector_save.py -d ${IMAGE_PATH} -t real -i 5
python full_vector_save.py -d ${IMAGE_PATH} -t semantic_annotations -i 5
