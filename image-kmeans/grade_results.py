#!/bin/python3
"""
Description:

Receiving arguments:
    1 - @required Path to csv to grade
    2 - @required Path to the csv containing the ids and labels
"""
import sys
import os
import pandas

from keras.applications.vgg19 import preprocess_input, VGG19

def main() -> int:
    model = VGG19()
    print(model.summary())

if __name__ == "__main__":
    sys.exit(main())
