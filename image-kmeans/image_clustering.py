#!/bin/python3
"""
Description:

Receiving arguments:
    1 - @required Path to folder containing the images to be clustered
    2 - @required Path to folder where the images and folders will be ordered
    3 - @required Path where the model is going to be loaded and/or saved

Author: peterglezroa
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas

from keras.preprocessing import image_dataset_from_directory 
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Constants
BATCHSIZE = 20

def printUsage():
    print(f"usage: {sys.argv[0]} <path to folder with images to be clustered> "\
        "<path to folder where everything will be ordered> "\
        "<path where the model will be loaded from>")

def main() -> int:
    if len(sys.argv) not 4:
        printUsage()
        return -1

    # TODO: check path and filetype for model

    # TODO: check folder with images

    # TODO: check that the other folder is created

    images = []

    # Read all images
    with os.scandir(path) as files:
        for file in files:
            # TODO: more image types
            if file.name.endswith(".png"):
                images.append(file.name)


    # TODO: load model

    # TODO: cluser images

    # TODO: symbolic links
    # os.symlink(src, dst, target_is_directory=False, *, dir_fd=None)

if __name__ == "__main__":
    sys.exit(main())
