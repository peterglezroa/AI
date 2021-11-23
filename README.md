# AI

## Image K-Means

### Problem
Imagine you are part of a big company that is in charge of administrating posts
with images like Pinterest, Instagram, Twitter, or Reddit. Every day you are
bombarded by thousand, maybe even millions of images that you have to check to
ensurre that all of them comply the terms and conditions of the application.

How can you have a fast response to take down images that are not allowed
without the utilization of all of your resources?

### Approach
The dimension of the problem would be drastically reduced if we previously have
clustered every post uploaded in the day in _n_ smaller groups of images that
share similarities.

_For example: we would not need to pay as much attention to
a group full of puppies phots than a group of human phots because in the later
it is more probable to have infringing images._

Therefore as my approach I'm going to experiment with clustering the images
with the use of a neural network and the k-means clustering algorithm.

#### Dataset
First we need to select a dataset from which will we test our hypothesis. For
this experiment i will use the
[flower dataset from kaggle](https://www.kaggle.com/olgabelitskaya/flower-color-images).

Also, we could have used a bigger dataset like Google's
[open image dataset](https://storage.googleapis.com/openimages/web/index.html).
There are several reasons as of why i will not use this dataset:
1. I tried it out for the first version of this model and was not able to
consistenly cluster images because of the bigger complexity. It is possible that
the model will be able to perform better with some improvements.
2. I do not have the hardware needed to work around with my limited time.

#### Model
Before we dive in to the implementation i would highly recommend to read
[Sumit Saha article that explains what is a Convolutional Neural Network](
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

As previously mentioned, we will use a CNN to highlight the features of each
image. For this model we will use an VGG19 architecture, which is one of the
most popular. This model consists of 19 layers:

1. Convolution layer
2. Convolution layer

![vgg19][vgg19]



### Results

### Improvements

#### Paper

#### Image Segmentation

### How to Run

### References
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

https://towardsdatascience.com/image-clustering-using-k-means-4a78478d2b83

vgg19 image: https://supportivy.com/comment-former-les-cnn-sur-imagenet/ 

[vgg19]: ./media/vgg19.jpg
