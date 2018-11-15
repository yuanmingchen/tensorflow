import tensorflow as tf
import numpy as np

# 初始化state为0
state = tf.Variable(0, dtype=tf.float64, name="counter")
print(state.name, state.value())
tensor = tf.constant([[1,2,3,],[4,5,6]])
print(tf.Session().run(tensor))
one = tf.constant(1, dtype=tf.float64)
two = tf.constant(2, dtype=tf.float64)

# 加一操作，计算的结果为new_value
new_value = tf.add(state, one)
new_value = tf.add(new_value, one)
new_value = tf.add(new_value, one)
new_value = tf.add(new_value, one)
new_value = tf.divide(new_value, two)
# 把new_value变量赋值给state
update = tf.assign(state, new_value)

# 如果定义了变量Variable，一定要调用下面的方法初始化变量！！！
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须调用此方法执行初始化
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
