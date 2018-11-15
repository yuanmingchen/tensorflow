import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
loss = 1
# 最基础的梯度下降优化器
tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 常用的几个高阶优化器
tf.train.AdamOptimizer(0.1)
tf.train.MomentumOptimizer(0.1)
tf.train.RMSPropOptimizer(0.1)
tf.train.AdagradOptimizer(0.1)