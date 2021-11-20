import numpy
import sys
import os
import matplotlib.pyplot as plt

from PIL import Image

COLS = 2
ROWS = 3

def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <image to transform>")

    trans = list()
    with Image.open(sys.argv[1]) as img:
        img = img.resize((224,224))

        trans.append(numpy.array(img))
        trans.append(numpy.array(img.rotate(45)))
        trans.append(numpy.array(img.transform(img.size, Image.EXTENT,
            data=[img.width/4, img.height/4, img.width*3/4, img.height*3/4])))

        m = 0.3
        xshift = abs(m)*img.width
        trans.append(img.transform(img.size, Image.AFFINE,
            [1, m, -xshift if m > 0 else 0,0,1,0], Image.BICUBIC))

        trans.append(numpy.array(img.transpose(method=Image.FLIP_LEFT_RIGHT)))
        trans.append(numpy.array(img.transpose(method=Image.FLIP_TOP_BOTTOM)))


    fig = plt.figure(figsize=(10, 8))
    for i, image in enumerate(trans):
        fig.add_subplot(ROWS, COLS, i+1)
        plt.imshow(image)
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
