import tensorflow as tf
import numpy as np


# 保存内容
def save():
    # 这里变量必须有name属性，以便存储起来之后可以根据name重新加载
    W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # 保存
    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, "my_net/save_net.ckpt")
        print("Save to path:", save_path)


# 把保存的内容提取出来
def restore():
    # 关键:提取的时候name必须和保存的时候一致！
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "my_net/save_net.ckpt")
        print("weights:", sess.run(W), "biases:", sess.run(b))


if __name__ == '__main__':
    # save();
    restore()
