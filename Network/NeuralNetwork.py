#!/usr/bin/env python
''''
Implement a Neural Network with n number of hidden layers
'''

## TensorFlow

## TensorFlow
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.contrib import rnn
import tensorflow as tf



__author__ = "Rahul Bhadani"
__copyright__ = "Copyright 2019, Rahul Bhadani, Arizona Board of Regents, The University of Arizona"
__credits__ = ["Rahul Bhadani", "Tensorflow"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Rahul Bhadani"
__email__ = "rahulbhadani@email.arizona.edu"
__status__ = "Under development"


class NLayerPerceptron:

    def __init__(self, input_dim = 784,  num_hidden_layer=2, hidden_layer_dimensions = [50, 50], output_dim = 10,
    toplevelscope="mainscope"):
        '''
        We first define the network architecture by passing network configuration through the constructor

        '''
        if len(hidden_layer_dimensions) != num_hidden_layer:
            raise ValueError("Number of hidden layer dimensions must match number of hidden layers")
        
        # Computational Graph for the Network
        self.graph = tf.Graph()
        self.session = tf.Session(graph = self.graph)

        with self.graph.as_default():
            self.input_dim = input_dim # Input Dimension
            self.num_hidden_layer = num_hidden_layer
            self.hidden_layer_dimensions = hidden_layer_dimensions
            self.out_dim = output_dim

            
            with tf.variable_scope(toplevelscope) as scope:
                self.scope_name = scope.name
                with tf.variable_scope("input") as scope:
                    # A Placeholder for holding training data
                    self.input = tf.placeholder(tf.float32, [None, input_dim], name="input")
                    
                    # A Placeholder for holding correct answer/label during training
                    self.labels = tf.placeholder(tf.float32, [None, output_dim], name="labels")
                    # The probability of a neuron being kept during drop_out
                    self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
                    
                # An empty placehold for weights
                self.weights = []
                self.biases = []
                self.activation = []
                self.dropout = []
                with tf.variable_scope("model") as scope:
                    with tf.variable_scope("layer_input") as scope:
                    ## Input Layer's weights, biases and activation function
                        w = tf.Variable(tf.truncated_normal([input_dim, hidden_layer_dimensions[0]], stddev= 0.1), name= "weights_input")
                        self.weights.append(w)
                        b = tf.Variable(tf.constant(0.1, shape=[hidden_layer_dimensions[0]]), name="biases_input")
                        self.biases.append(b)
                        a = tf.nn.relu(tf.matmul(self.input, w) + b)
                        self.activation.append(a)
                        d= tf.nn.dropout(a, self.keep_prob)
                        self.dropout.append(d)

                    # Intermediate Hidden layers
                    for i  in range(0, num_hidden_layer):
                        with tf.variable_scope("layer_"+str(i)) as scope:
                            w = tf.Variable(tf.truncated_normal([hidden_layer_dimensions[i-1], hidden_layer_dimensions[i]], stddev= 0.1), name= "weights_"+str(i))
                            self.weights.append(w)
                        
                            b= tf.Variable(tf.constant(0.1, shape=[hidden_layer_dimensions[i]]), name="biases_"+str(i))
                            self.biases.append(b)
                            a = tf.nn.relu(tf.matmul(self.dropout[i-1],  w) + b)
                            self.activation.append(a)
                            d = tf.nn.dropout(a, self.keep_prob)
                            self.dropout.append(d)
                    

                    #Final Layer
                    with tf.variable_scope("layer_output") as scope:
                        w = tf.Variable(tf.truncated_normal([hidden_layer_dimensions[num_hidden_layer-1], output_dim], stddev=0.1) , name = "weights_output" )
                        self.weights.append(w)
                        b = tf.Variable(tf.constant(0.1, shape=[output_dim]), name = "biases_output")
                        self.biases.append(b)
                        a  =tf.matmul(self.dropout[num_hidden_layer], w ) + b
                        self.activation.append(a)

                self.init_op = tf.global_variables_initializer()
                self.saver = tf.train.Saver()
                self.variables = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_name)
                self.init_vars = tf.variables_initializer(self.variables)

            with tf.variable_scope(self.scope_name):
                v = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
                for vi in v:
                    print(vi)
        self.training_accuracy = []
        self.testing_accuracy = []


    def train(self, data, optimizer, learning_rate = 0.01, keep_prob=0.6, num_epoch = 16, batch_size = 128):

        with tf.variable_scope(self.scope_name) as scope:
            print("Scope Name: ", self.scope_name)
            v = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
            for vi in v:
                print(vi)

        with self.graph.as_default():
            with tf.variable_scope(self.scope_name) as scope:

                loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.activation[len(self.activation)-1])
                print("Loss = ", loss)
                train_step = optimizer(learning_rate).minimize(loss)

                correct_prediction = tf.equal(tf.argmax(self.activation[len(self.activation)-1], 1), tf.argmax(self.labels, 1))

                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.summary.scalar("Accuracy", accuracy)
            tf.summary.scalar("Loss", tf.reduce_mean(loss))

            summary_op = tf.summary.merge_all()
            init_train = tf.global_variables_initializer()

            with self.session as sess:
                sess.run(self.init_op)
                sess.run(init_train)
                #sess.run(self.init_vars)
                # Train the network
                for step in range(num_epoch):
                    average_accuracy = 0.0
                    average_loss = 0.0
                    total_batch = int(data.train.num_examples/batch_size)
                    for i in range(total_batch):
                        batch_x, batch_y = data.train.next_batch(batch_size)
                        feed_dict = {self.input:batch_x, self.labels: batch_y, self.keep_prob: keep_prob}
                        sess.run([train_step, loss], feed_dict)
                        feed_dict = {self.input:batch_x, self.labels: batch_y, self.keep_prob: 1.0}
                        acc = sess.run(accuracy,  feed_dict)
                        average_accuracy += acc/total_batch
                        #print("Mid Train Accuracy is : ", acc, " at step: ", step )
                    if(step % 4 == 0):
                        print("Epoch:", '%05d' % (step+1), "Average accuracy =","{:.8f}".format(average_accuracy))
                
                # Training Accuracy
                feed_dict = {self.input:data.train.images, self.labels: data.train.labels, self.keep_prob: 1.0}
                acc = sess.run(accuracy,  feed_dict)
                print("Final Training Accuracy:", acc)

                # Testing Accuracy
                feed_dict = {self.input:data.test.images, self.labels: data.test.labels, self.keep_prob: 1.0}
                acc = sess.run(accuracy,  feed_dict)
                print("Final Testing Accuracy:", acc)

