# -*- coding:utf-8 -*-
# placeholder：每次想使用不同的值，执行的时候外部传入值来填充placeholder
import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)  # 还可以传入一个shape规定形状
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # 运行的时候使用字典feed_dict传入需要的值（填充placeholder）
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
