#!/usr/bin/env python
''''
Implement Affine transformation
'''


__author__ = "Rahul Bhadani"
__copyright__ = "Copyright 2019, Rahul Bhadani, Arizona Board of Regents, The University of Arizona"
__credits__ = ["Rahul Bhadani", "Tensorflow"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Rahul Bhadani"
__email__ = "rahulbhadani@email.arizona.edu"
__status__ = "Under development"

import tensorflow as tf

x = tf.placeholder(tf.float32, [3, 1], name="input")
A = tf.Variable( [[1, 2, 3], [4, 5, 6], [7, 8, 9]] , dtype=tf.float32,  name = "A")
b = tf.Variable(tf.random.normal([3, 1]), name = "b")

y = tf.matmul(A, x) + b

print(A)
print(x)
print(b)
print(y)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {x: [[1], [2], [3]]}
    A_val = sess.run(A)
    b_val = sess.run(b)
    y_val = sess.run(y, feed_dict)

    print('A = ' , A_val)
    print('b = ', b_val)
    print('c = ', y_val)

g = tf.Graph()
with g.as_default():
    xg = tf.placeholder(tf.float32, [3, 1])
    Ag = tf.Variable( [[1, 2, 3], [4, 5, 6], [7, 8, 9]] , dtype=tf.float32)
    bg = tf.Variable(tf.random.normal([3, 1]))
    yg = tf.matmul(Ag, xg) + bg
    init_op = tf.global_variables_initializer()

print(Ag)
print(xg)
print(bg)
print(yg)

with tf.Session(graph = g) as sess:
    sess.run(init_op)
    feed_dict = {xg: [[1], [2], [3]]}
    A_val = sess.run(Ag)
    b_val = sess.run(bg)
    y_val = sess.run(yg, feed_dict)

print('A = ' , A_val)
print('b = ', b_val)
print('c = ', y_val)