# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""

from __future__ import print_function
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #Disable Tensorflow debugging information
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Set GPU device -1
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # filter: [filter_height, filter_width, in_channels, out_channels]
    # strides=[batch_size, 高度, 宽度, 通道数]，batch_size和 通道数对应的步长必须为1
    # padding='SAME'结果与原图片大小一样
    # x是图片的所有信息，W是本卷积层的权重
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # 最大池化层
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


time1 = time.time()
with tf.device('/gpu:0'):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1代表图片个数任意*高度*宽度*最后的1代表输入通道数

    # conv1 layer
    W_conv1 = weight_variable([5, 5, 1, 32])  # filter:5*5   输入通道数：1  输出通道数：32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出大小：28*28*32（因为padding设置为same且stride=1，所以宽高不变）
    h_pool1 = max_pool_2x2(h_conv1)  # 输出大小：14*14*32（因为padding设置为same且stride=2，所以宽高减半）
    # conv2 layer
    W_conv2 = weight_variable([5, 5, 32, 64])  # filter:5*5   输入通道数：32  输出通道数：64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 输出大小：14*14*64（因为padding设置为same且stride=1，所以宽高不变）
    h_pool2 = max_pool_2x2(h_conv2)  # 输出大小：7*7*64
    # func1 layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # func2 layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.Session()
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    time2 = time.time()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print(compute_accuracy(
                mnist.test.images[:1000], mnist.test.labels[:1000]))
time3 = time.time()
print("建立网络阶段用时%.5f秒" % (time2-time1), "训练阶段用时%.5f秒" % (time3-time2), "总共用时%.5f秒" % (time3 - time1))
