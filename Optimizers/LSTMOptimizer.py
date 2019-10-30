#!/usr/bin/env python

# This class defines the API to add Ops to train a model. 
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import gradients
import tensorflow as tf

import numpy as np
import math
import os

class BaseLSTMOptimizer:
    """
    Optimizer that implements LSTM to optimize
    and Optimizee network
    """

    def weight_initializer(self):
        return tf.truncated_normal_initializer(stddev=0.1)

    def bias_initalizer(self):
        return tf.constant_initializer(0.0)

    def _get_variable(self, getter, name, *args, **kwargs):
        kwargs['trainable'] = self.is_training
        if not self.is_training:
            kwargs['collections'] = [tf.GraphKeys.MODEL_VARIABLES]
        return getter(name, *args, **kwargs)

    # Placeholder
    def ph(self, shape, dtype=tf.float32):
        return tf.placeholder(dtype, shape=shape)

    # Define a function to create a fully connected layer
    # CHECK: Is c dimension of the hidden layer?
    def fc(self, x, c, use_bias = True):
        '''
        x is input to this fully connected Layer
        This function basically doing all of this:

        w = tf.Variable(tf.truncated_normal([input_dim, hidden_layer_dimensions[0]], stddev= 0.1), name= "weights_input")
        self.weights.append(w)
        b = tf.Variable(tf.constant(0.1, shape=[hidden_layer_dimensions[0]]), name="biases_input")
        self.biases.append(b)
        a = tf.matmul(self.input, w) + b

        '''
        n = x.get_shape()[1]
        w = tf.get_variable("w", [n, c], initializer=self.weight_initializer())
        if use_bias:
            b = tf.get_variable("b", [c], initializer=self.bias_initalizer())
            return tf.matmul(x, w) + b
        else:
            return tf.matmul(x, w)



    def __init__(self, steps_to_unroll,  optimizee=None, n_bptt_steps = None,  learning_rate = 1e-4, use_avg_loss=False,  is_training=True,  optimizer_name = 'Adam', use_locking = False, name="BaseLSTMOptimizer"):
        '''
        optimizee is the actual input data
        '''
        super(BaseLSTMOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._steps_to_unroll = steps_to_unroll
        self.is_training = is_training
        self.kwargs = kwargs
        if self.is_training:
            self.optimizee = optimizee
            self.n_bptt_steps = n_bptt_steps
            self.optimizer_name = optimizer_name
            self.x_dim = optimizee.get_x_dim()
            self.use_avg_loss = use_avg_loss

        self.session = tf.get_default_session()
        

    def _get_los_and_grad(self, f, i, x):
        if isinstance(f, list):
            return f[0], f[1]
        loss = f(i, x)
        grad = gradients.gradients(loss, x)[0]
        return loss, grad
        
    def _log_encode(self, x, p = 10.0):
        xa = tf.log(tf.maximum(tf.abs(x), math.exp(-p)))/p
        xb = tf.clip_by_value(x*math.exp(p), -1, 1)
        return tf.pack([xa, xb], 1)


    # Private Function to create placeholder for input to the layer
    def _build_input(self):
        self.x = tf.placeholder(tf.float32, shape=[self.x_dim])
        self.input_state = [self.x]

    # Private Function to create initial state
    def _build_inital(self):
        self.initial_state = [self.x]

    def _build_loop(self):
        with tf.variable_scope("loop") as self.loop_scope:
            self.states = []

            state = self.input_state

            self.all_internal_loss = []
            
            for i in range(self.n_bptt_steps):
                state, loss = self._iter(self.f, i, state)
                self.states.append(state)
                self.all_internal_loss.append(loss)

                if i == 0:
                    self.loop_scope.reuse_variables()

        def _build_loss(self):
            if self.use_avg_loss:
                self.loss = tf.reduce_mean(self.all_internal_loss)
            else:
                self.loss = self.all_internal_loss[-1]
        
        def _build_optimizer(self):
            if self.optimizer_name == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self._learning_rate)
            if self.optimizer_name == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(self._learning_rate)
            if self.optimizer_name == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(self._learning_rate, self.kwargs['beta'])
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.all_vars)
            self.apply_gradients = self.optimizer.apply_gradients(self.gradients)
        
        def _build(self):
            with tf.variable_scope("nn_opt", custom_getter=self._get_variable) as scope:
                self.summary_writer = tf.summary.FileWriter(self.name+"_data", self.session.graph)
                self.summaries = []

                self._build_input()
                self._build_inital()

                if self.is_training:
