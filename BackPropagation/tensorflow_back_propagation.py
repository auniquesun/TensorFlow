import numpy as np
import tensorflow as tf
from random import uniform
from sklearn import preprocessing

def sigmoidprime(x):
    return tf.multiply(tf.sigmoid(x), tf.subtract(1.0, tf.sigmoid(x)))

# 构造训练数据、测试数据
X = np.zeros((20000, 3))    
y = np.zeros((20000, 2))

for i in range(20000):
    if i%2 == 1:
        X[i, 0], X[i, 1], X[i, 2] = uniform(70.0, 100.0), uniform(70.0, 100.0), uniform(70.0, 100.0)
        y[i, 0], y[i, 1] = 1., .0

    else:
        X[i, 0], X[i, 1], X[i, 2] = uniform(0.0, 30.0), uniform(0.0, 30.0), uniform(0.0, 30.0)
        y[i, 0], y[i, 1] = 0., 1.

X_test = np.zeros((5000, 3))
y_test = np.zeros((5000, 2))

for i in range(5000):
    if i%2 == 1:
        X_test[i, 0], X_test[i, 1], X_test[i, 2] = uniform(65.0, 89.0), uniform(60.0, 110.0), uniform(55.0, 76.0)
        y_test[i, 0], y_test[i, 1] = 1., .0

    else:
        X_test[i, 0], X_test[i, 1], X_test[i, 2] = uniform(0.0, 39.0), uniform(10.0, 49.0), uniform(-5, 51.0)
        y_test[i, 0], y_test[i, 1] = 0., 1.

preprocessing.scale(X)
preprocessing.scale(X_test)

a0 = tf.placeholder(tf.float32, shape=(None, 3))    #((None, 3))
y0 = tf.placeholder(tf.float32, shape=(None, 2))    #((None, 2))

w1 = tf.Variable(tf.random_uniform([3,4]), dtype=tf.float32)    
b1 = tf.Variable(tf.random_uniform([4]), dtype=tf.float32)
z1 = tf.add(tf.matmul(a0, w1), b1)  # z1的形状  (None,4)
a1 = tf.sigmoid(z1)

w2 = tf.Variable(tf.random_uniform([4,2]), dtype=tf.float32)
b2 = tf.Variable(tf.random_uniform([2]), dtype=tf.float32)
z2 = tf.add(tf.matmul(a1, w2), b2)
a2 = tf.sigmoid(z2)


delta_a2 = a2 - y0
delta_z2 = tf.multiply(delta_a2, sigmoidprime(z2))
delta_b2 = delta_z2
delta_w2 = tf.matmul(tf.transpose(a1), delta_z2)

delta_a1 = tf.matmul(delta_z2, tf.transpose(w2))
delta_z1 = tf.multiply(delta_a1, sigmoidprime(z1))
delta_b1 = delta_z1
delta_w1 = tf.matmul(tf.transpose(a0), delta_z1)

eta = tf.constant(0.5)
step = [
    tf.assign(w1, tf.subtract(w1, tf.multiply(eta, delta_w1))),
    tf.assign(b1, tf.subtract(b1, tf.multiply(eta, delta_b1))),
    tf.assign(w2, tf.subtract(w2, tf.multiply(eta, delta_w2))),
    tf.assign(b2, tf.subtract(b2, tf.multiply(eta, delta_b2)))
]

acct_mat = tf.equal(tf.argmax(a2, 1), tf.argmax(y0, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

count = 0
batch_size = 200
iterations = 10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(iterations):
        batch_x, batch_y = X[count%100 * batch_size:(i+1)*batch_size, :], y[count%100 * batch_size:(i+1)*batch_size, :]
        count += 1
        sess.run(step, feed_dict = {a0: batch_x, y0: batch_y})

        if iteration%1000 == 0:
            res = sess.run(acct_res, feed_dict={a0: X_test[:, :], y0: y_test[:, :]})
            print(iteration, ':', res)
