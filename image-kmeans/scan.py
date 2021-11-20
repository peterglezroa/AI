#!/bin/python3
"""
Description:

Receiving arguments:
    1 - @required Path on where to save/load the model
    2 - @required Path where the images are stored

Author: peterglezroa
"""
import sys
import os
import numpy
import matplotlib.pyplot as plt

from pillow import Image

from keras.preprocessing.image import load_img
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils import make_csv, view_cluster

# 1012 images
LEARNING_RATE = 0.001
BATCH_SIZE = 200
IMAGE_SIZE = (224, 224)
MIN_CLUSTERS = 3
MAX_CLUSTERS = 20

def preprocess_image(image: str) -> numpy.array:
    img = numpy.array(load_img(image, target_size=(224,224)))
    return preprocess_input(img.reshape(224, 224, 3))

def image_variation_training(model: Model, image: str) -> numpy.array:
    """
    Custom training loop to improve according to the distance between variations
    of a same image
    """
    img = Image.open(image)
    inpt = [
        preprocess_input(image.reshape(224,244,3)),
    ]

def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <path where to save/load model> "\
            "<path to folder including the labeled images>")
        return -1

    # Scan directory
    images = list()
    with os.scandir(sys.argv[2]) as files:
        for file in files:
            images.append(file.name)
    images = numpy.array(images)

    # Preprocess images
    pwd = os.getcwd()
    os.chdir(sys.argv[2])
    imgs = numpy.array([preprocess_image(img) for img in images[:BATCH_SIZE]])
    print(numpy.shape(imgs))
    os.chdir(pwd)

    # Copy vgg16 model without initial weights and remove the softmax layer
    model = Sequential()
    model.add(Conv2D(input_shape(224,244,3), filters=64, kernel_size=(3,3),
        padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))

    model.compile(
        optimizer = Adam(lr=LEARNING_RATE),
        loss = keras.lossed.categorical_crossentropy, metrics=["accuracy"]
    )

    print(model.summary())

    # Checkpoint that saves to memory in case it improved from the previous version
    checkpoint = ModelCheckpoint("scan_s1.h5", monitor="val_acc", verbose=1,
        save_best_only=True, save_weights_only=False, mode="auto", period=1)

    hist = model.fit_generator(
        epochs = 100,
        steps_per_epoch = 100,
        generator=traindata,
        validation_data=testdata,
        callbacks=[checkpoint, early]
    )
    
if __name__ == "__main__":
    sys.exit(main())
