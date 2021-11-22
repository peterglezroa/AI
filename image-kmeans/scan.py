#!/bin/python3
"""
Description:

Receiving arguments:
    1 - @required Path on where to save/load the model
    2 - @required Path where the images are stored

Author: peterglezroa
"""
import sys, os, numpy, matplotlib.pyplot as plt
from PIL import Image
from random import sample


import keras.backend as K
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.callbacks import History
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Model, Sequential
from keras.preprocessing.image import load_img

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils import make_csv, view_cluster

# 1012 images
EPOCHS = 1
BATCH_SIZE = 10
MIN_CLUSTERS = 3
MAX_CLUSTERS = 20

TRANSFORMATIONS = 6 

def preprocess_image(image) -> numpy.array:
    img = numpy.array(load_img(image, target_size=(224,224)))
    return preprocess_input(img.reshape(224, 224, 3))

def image_transformations(file_name: str) -> numpy.array:
    """ Transforms an image to 6 different transformations """
    trans = list()
    with Image.open(file_name) as img:
        img = img.resize((224,224))

        # Normal image
        trans.append(numpy.array(img))

        # 45 degree rotation
        trans.append(numpy.array(img.rotate(45)))

        # Crop image
        trans.append(numpy.array(img.transform(img.size, Image.EXTENT,
            data=[img.width/4, img.height/4, img.width*3/4, img.height*3/4])))

        # Wierd transformation I found in the internet
        m = 0.3
        xshift = abs(m)*img.width
        trans.append(numpy.array(img.transform(img.size, Image.AFFINE,
            [1, m, -xshift if m > 0 else 0,0,1,0], Image.BICUBIC)))

        # Flip l->r image
        trans.append(numpy.array(img.transpose(method=Image.FLIP_LEFT_RIGHT)))

        # Flip t->b image
        trans.append(numpy.array(img.transpose(method=Image.FLIP_TOP_BOTTOM)))
    return numpy.array(trans)[..., :3]

def distance_between_predictions(y_true, y_pred):
#    mask = K.mean(y_true)
    center = K.mean(y_pred, axis=0)
    K.print_tensor(y_pred, message="ypred")
#    K.print_tensor(K.std(y_pred), message="std")
    #K.print_tensor(mask, message="mask")
    #K.print_tensor((1-mask)*K.std(y_pred, axis=0) - mask*K.std(y_pred, axis=0), message="loss")
    #return (1-mask)*K.std(y_pred, axis=0) - mask*K.std(y_pred, axis=0)
    return center - y_pred[0]

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

    # Copy vgg16 model without initial weights and remove the softmax layer
    """
    model = Sequential([
        Conv2D(input_shape=[224,244,3], filters=64, kernel_size=(3,3),
                padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),

        Flatten(),
        Dense(units=4096, activation="relu"),
        Dense(units=4096, activation="relu"),
        Dense(units=100, activation="relu"),
    ])
    """
    initial_model = VGG19()
    initial_model = Model(inputs=initial_model.inputs, outputs=initial_model.layers[-2].output)
    model = Model(
        initial_model.input,
        Dense(units=100, activation="relu")(initial_model.output)
    )

    model.compile(
        loss = distance_between_predictions,
        optimizer = "adam",
        metrics=["accuracy"]
    )
    print(model.summary())
    history = History()

    # Go to folder with images for easier implementation
    pwd = os.getcwd()
    os.chdir(sys.argv[2])

    zeros = numpy.zeros([TRANSFORMATIONS, 100])
    ones = numpy.ones([BATCH_SIZE, 100])

    for epoc in range(0, EPOCHS):
        print(f"\nOuter epoch #{epoc}")
        img_sample = sample(images, BATCH_SIZE)

        # Preprocess samples
        pro_sample = list()
        for image in img_sample:
            pro_sample.append(preprocess_image(image))
        np_sample = numpy.array(pro_sample)

        for image in img_sample:
            # Train for min dist between transformations of an image
            inpts = preprocess_input(image_transformations(image))
            model.fit(inpts, zeros, callbacks=[history])

            # Train where we penalize low entropy
#            model.fit(np_sample, ones)
        print(history.history["loss"])

    # Return to working directory
    os.chdir(pwd)



    # Checkpoint that saves to memory in case it improved from the previous version

    """
    checkpoint = ModelCheckpoint("scan_s1.h5", monitor="val_acc", verbose=1,
        save_best_only=True, save_weights_only=False, mode="auto", period=1)

    hist = model.fit_generator(
        epochs = 100,
        steps_per_epoch = 100,
        generator=traindata,
        validation_data=testdata,
        callbacks=[checkpoint, early]
    )
    """
    
if __name__ == "__main__":
    sys.exit(main())
