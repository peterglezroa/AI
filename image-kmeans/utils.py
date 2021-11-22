import pandas
import os
import numpy
import matplotlib.pyplot as plt

from PIL import Image

from keras.preprocessing.image import load_img

COLS = 7
ROWS = 7

def make_csv(file_name: str, ids: numpy.array, clusters: numpy.array):
    """ Make csv using pandas. """
    df = pandas.DataFrame(data={"id": ids, "cluster": clusters})
    df.to_csv(file_name, index=False)

def view_cluster(ids:numpy.array, clusters:numpy.array, cid:int=0):
    fig = plt.figure(figsize=(10, 10))
    limit = 1
    for image, cluster in zip(ids, clusters):
        if cluster == cid and limit <= ROWS*COLS:
            img = numpy.array(load_img(image))
            fig.add_subplot(ROWS, COLS, limit)
            plt.imshow(img)
            plt.axis("off")
            limit += 1
    plt.show()
