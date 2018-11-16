import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

with tf.device('/gpu:0'):
    def add_layer(inputs, in_size, out_size, activation_function=None):
        with tf.name_scope('layer'):
            with tf.name_scope('weight'):
                # 用tf.random_normal生成随机变量作为初始值，比全为0要好很多
                weights = tf.Variable(tf.random_normal([in_size, out_size]))
            with tf.name_scope('biases'):
                # 推荐初始值不为0，所有初始为0.1
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, weights) + biases
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                # 激活函数默认有名字，可以不写
                outputs = activation_function(Wx_plus_b)
            return outputs


    def compute_accuracy(v_xs, v_ys):
        global preediction
        y_pre = sess.run(prediction, feed_dict={xs: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
        return result

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
    ys = tf.placeholder(tf.float32, [None, 10])

    prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)
    max_accuracy = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for step in range(50000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
            # if step % 50 == 0:
            loss = sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys})
            accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                print('step:', step, 'loss:', loss, 'accuracy:', accuracy)
        print("最高准确率", max_accuracy)
