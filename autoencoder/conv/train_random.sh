## Traing AutoEncoder

# Image path
IMAGE_PATH='../../data_processing/'

echo 'conv autoencoder training..'
python train_random.py -d ${IMAGE_PATH} -t real -i 5 -n 15000
python train_random.py -d ${IMAGE_PATH} -t semantic_annotations -i 5 -n 15000

echo 'vector save..'
python vector_save.py -d ${IMAGE_PATH} -t real -i 5
python vector_save.py -d ${IMAGE_PATH} -t semantic_annotations -i 5

#echo 'full vector save..'
#python full_vector_save.py -d ${IMAGE_PATH} -t real -i 30
#python full_vector_save.py -d ${IMAGE_PATH} -t semantic_annotations -i 30
