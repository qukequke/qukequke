import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


k = 2
x = np.linspace(0, 1, 100)
noise = np.random.rand(1, 100)
y = k * x ** 4 + noise
fig = plt.figure()
plt.scatter(x, y)
x = x.astype('float32')
y = y.astype('float32')
x = x.reshape(1, 100)
y = y.reshape(1, 100)
# plt.show()


X = tf.placeholder(dtype=tf.float32, shape=[1, 100], name='x')
Y = tf.placeholder(dtype=tf.float32, shape=[1, 100], name='y')

w1 = tf.get_variable(dtype=tf.float32, name='w1', shape=[100, 100], initializer=tf.zeros_initializer)
w2 = tf.get_variable(dtype=tf.float32, name='w2', shape=[100, 100], initializer=tf.zeros_initializer)

b1 = tf.get_variable(dtype=tf.float32, name='b1', shape=[100], initializer=tf.zeros_initializer)
b2 = tf.get_variable(dtype=tf.float32, name='b2', shape=[100], initializer=tf.zeros_initializer)

z1 = tf.matmul(X, w1) + b1
a1 = tf.nn.relu(z1)

z2 = tf.matmul(a1, w2) + b2

entropy = tf.square(z1-Y)
loss = tf.reduce_mean(entropy)

optimer = tf.train.AdamOptimizer(1e-3).minimize(loss)

y_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(optimer, feed_dict={X: x, Y: y})
        print(sess.run(loss, feed_dict = {X: x, Y: y}))
    y_pre = (sess.run(z1, feed_dict={X:x}))
    print(y_pre.shape)
    # print(np.array(y_list).reshape(1, 100))
    plt.scatter(x.reshape(-1), y_pre.reshape(-1), s=6)
    plt.show()


