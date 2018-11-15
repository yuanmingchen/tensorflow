import tensorflow as tf
import numpy as np
import matplotlib as mlp

mlp.use('TkAgg')
import matplotlib.pyplot as plt
import pylab


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            # 用tf.random_normal生成随机变量作为初始值，比全为0要好很多
            weights = tf.Variable(tf.random_normal([in_size, out_size]))
        tf.summary.histogram("weights1", weights)
        with tf.name_scope('biases'):
            # 推荐初始值不为0，所有初始为0.1
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        tf.summary.histogram("biases1", biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            # 激活函数默认有名字，可以不写
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram("output1", outputs)
        return outputs


# 新建一些数据并增加一个维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 噪声，期望为0，标准差为0.5
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, shape=[None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, shape=[None, 1], name='y_input')
# relu激活函数名字默认为relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction, name='square')
                                        , reduction_indices=[1], name='sum'), name='mean')
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir="logs", graph=sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
for step in range(1000):
    sess.run(train_step,feed_dict={xs: x_data, ys: y_data})
    if step % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, step)