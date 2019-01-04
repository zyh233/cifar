# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:38:44 2019

@author: zhangyonghong
"""

import tensorflow as tf
#import numpy as np
#import os

def get_w(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_b(shape):
    #b = tf.constant(0.01, shape = shape)
    return tf.Variable(tf.zeros(shape))

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

def forward(x, keep_prob, regularizer):
    
    x_image = tf.reshape(x, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image, [0, 2, 3, 1])
    #x_image = tf.reshape(x, [-1, 32, 32, 3])
    w_conv1 = get_w([5,5,3,32], regularizer)
    b_conv1 = get_b([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2_2(h_conv1)
    
    
    w_conv2 = get_w([5,5,32,64], regularizer)
    b_conv2 = get_b([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2_2(h_conv2)
    
    
    w_fc1 = get_w([8*8*64, 1024], regularizer)
    b_fc1 = get_b([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1,8*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    w_fc2 = get_w([1024, 10], regularizer)
    b_fc2 = get_b([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
    
    return y
    