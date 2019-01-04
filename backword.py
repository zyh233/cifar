# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 19:44:37 2019

@author: zhangyonghong
"""

import tensorflow as tf
#import numpy as np
import os
import forward
import readData

batch_size = 50
regularizer = 0.0001
steps = 5000
model_path = "./model/"
model_name = "cifar_model"

def next_batch(x_train, y_train, start):
    #start = np.random.randint(0, 49999-batch_size)
    end = start + batch_size
    xs = x_train[start:end, :]/255
    ys = y_train[start:end, :]
    return xs, ys

def backword(x_train, y_train, x_test, y_test):
    
    x = tf.placeholder(tf.float32, [None, 3072], name = 'x')
    y_ = tf.placeholder(tf.float32, [None, 10], name = 'y')
    keep_prob = tf.placeholder(tf.float32)
    y = forward.forward(x, keep_prob, regularizer)
    
    #cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    #loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1])
    #tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y, name='cross_entropy_per_example')
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('losses'))
    
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(steps):
            start = i*batch_size%50000
            xs, ys = next_batch(x_train, y_train, start)
            sess.run(train_step, feed_dict={x: xs, y_: ys, keep_prob: 0.5})
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: xs, y_: ys, keep_prob: 1.0})
                print("step %d, train accuracy %g"%(i, train_accuracy))
        saver.save(sess, os.path.join(model_path, model_name))
        test_accuracy = accuracy.eval(feed_dict = {x: x_test[1:1000,:]/255, y_: y_test[1:1000,:], keep_prob: 1.0})
        print("test accuracy %g" % test_accuracy)
        
def main():
    [x_train, y_train, x_test, y_test] = readData.readData()
    backword(x_train, y_train, x_test, y_test)
    
if __name__ == '__main__':
    main()
    
    
    