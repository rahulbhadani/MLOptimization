#!/usr/bin/env python
''''
Implement a Neural Network with n number of hidden layers
'''

##  My Import
from NeuralNetwork import NLayerPerceptron
import sys
import warnings

import tensorflow as tf
#It will download and read in the data automatically
from tensorflow.examples.tutorials.mnist import input_data


if not sys.warnoptions:
    warnings.simplefilter("ignore")

__author__ = "Rahul Bhadani"
__copyright__ = "Copyright 2019, Rahul Bhadani, Arizona Board of Regents, The University of Arizona"
__credits__ = ["Rahul Bhadani", "Tensorflow"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Rahul Bhadani"
__email__ = "rahulbhadani@email.arizona.edu"
__status__ = "Under development"


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batchSize = 512
batch_x, batch_y = mnist.train.next_batch(batchSize)

optimizer = tf.train.AdamOptimizer

N = NLayerPerceptron()
N.train(x= batch_x, y = batch_y, optimizer=optimizer)