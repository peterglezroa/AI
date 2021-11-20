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

def image_transformations(file_name: str) -> numpy.array:
    trans = list()
    with Image.open(file_name) as img:
        img = img.resize((224,224))

        trans.append(numpy.array(img))
        trans.append(numpy.array(img.rotate(45)))
        trans.append(numpy.array(img.transform(img.size, Image.EXTENT,
            data=[img.width/4, img.height/4, img.width*3/4, img.height*3/4])))

        m = 0.3
        xshift = abs(m)*img.width
        trans.append(numpy.array(img.transform(img.size, Image.AFFINE,
            [1, m, -xshift if m > 0 else 0,0,1,0], Image.BICUBIC)))

        trans.append(numpy.array(img.transpose(method=Image.FLIP_LEFT_RIGHT)))
        trans.append(numpy.array(img.transpose(method=Image.FLIP_TOP_BOTTOM)))
    return numpy.array(trans)
