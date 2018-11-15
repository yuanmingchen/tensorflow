import tensorflow as tf
import numpy as np
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import pylab

def add_layer(inputs, in_size, out_size, activation_function=None):
    # 用tf.random_normal生成随机变量作为初始值，比全为0要好很多
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 推荐初始值不为0，所有初始为0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 新建一些数据并增加一个维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 噪声，期望为0，标准差为0.5
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, shape=[None, 1])
ys = tf.placeholder(tf.float32, shape=[None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    # add_subplot(row,column,num),三个参数分别表示图标的行数、列数、和这是第几个图。
    ax = fig.add_subplot(1, 1, 1)
    # 增加一些散点
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    for step in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if step % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                # 删除lines中的第一条线，实际上我们只有一条线
                ax.lines.remove(lines[0])
            except Exception as e:
                print(e)
            prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
            # 画一条线,横坐标是x_data,纵坐标是prediction_value，颜色是红色，线的宽度为5
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            # 暂停0.1s
            plt.pause(0.1)
            pylab.show()
plt.ioff()
plt.show()