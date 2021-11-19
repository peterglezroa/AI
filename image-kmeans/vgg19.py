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

from keras.preprocessing.image import load_img
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1012 images
BATCH_SIZE = 200
IMAGE_SIZE = (224, 224)
MIN_CLUSTERS = 3
MAX_CLUSTERS = 20

def preprocess_image(image) -> numpy.array:
    img = numpy.array(load_img(image, target_size=(224,224)))
    return preprocess_input(img.reshape(224, 224, 3))

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

    # Preprocess images
    os.chdir(sys.argv[2])
    imgs = numpy.array([preprocess_image(img) for img in images[:BATCH_SIZE]])
    print(numpy.shape(imgs))

    # Copy vgg19 model without the softmax layer
    model = VGG19()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    features = model.predict(imgs, use_multiprocessing=True)

    # Reduce the amount of dimensions with PCA
    pca = PCA(n_components=150)
    pca.fit(features)
    final_features = pca.transform(features)

    # Cluster
    kmeans = list()
    rnge = list(range(MIN_CLUSTERS, MAX_CLUSTERS))
    for k in range(MIN_CLUSTERS, MAX_CLUSTERS):
        clusters = KMeans(n_clusters=k, random_state=22)
        clusters.fit(final_features)

        kmeans.append(clusters.inertia_)

    # Plot to see clusters
    plt.figure(figsize=(10, 10))
    plt.plot(rnge, kmeans)
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
