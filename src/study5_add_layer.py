import tensorflow as tf
import numpy as np


def add_layer(inputs,in_size,out_size,activation_function = None):
    # 用tf.random_normal生成随机变量作为初始值，比全为0要好很多
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    # 推荐初始值不为0，所有初始为0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,weights) + biases
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs