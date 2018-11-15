import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    matrix1 = tf.constant([[3,3]])
    matrix2 = tf.constant([[2], [2]])
    product = tf.matmul(matrix1,matrix2)

    # 方法1
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()

    # 方法2,系统自动执行close
    with tf.Session() as sess:
        result2 = sess.run(product)
        print(result2)

