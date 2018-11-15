import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 开始搭建tensorflow框架
# tf.Variable参数变量,初始化权重为-0.1到0.1的一个随机数，[1]表示输出的形状
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 初始化偏置为0
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases
# loss = (y-y_data)²/n
loss = tf.reduce_mean(tf.square(y - y_data))
# 选择梯度下降优化器，学习率设置为0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 传入损失函数，目标是最小化目标函数
train = optimizer.minimize(loss)
# 初始化所有参数变量
init = tf.initialize_all_variables()
# 搭建框架结构结束

# 训练前的准备工作
# 定义一个运行环境Session
sess = tf.Session()
# 激活整个神经网络，即执行初始化
sess.run(init)

# 开始训练
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
# 关闭session
sess.close()
