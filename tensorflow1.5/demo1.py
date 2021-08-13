import tensorflow as tf
import numpy as np

import  matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc", size=14)

#使用tensorflow做线性回归
def line_mode():
    x_data = np.linspace(-.9, .9, 200, dtype=float)
    y_data = x_data*.5 + 1

    k = tf.Variable(0.0, dtype=tf.float32)
    b = tf.Variable(0.0, dtype=tf.float32)

    x = tf.compat.v1.placeholder(tf.float32,shape=(None))
    y = tf.compat.v1.placeholder(tf.float32,shape=(None))

    y_pred = x*k + b
    loss = tf.reduce_mean(tf.square(y - y_pred))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.02).minimize(loss)

    init = tf.compat.v1.initializers.global_variables()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for step in range(0, 300):
            sess.run(optimizer, feed_dict={x:x_data, y:y_data})
            if step % 20 == 0:
                print("k:{},b:{}".format(sess.run(k), sess.run(b)))


#简单的回归神经网络，非线性回归
def simple_net_reg():
    x_data = np.linspace(-.9, .9, 200, dtype=float)
    nouse = np.random.normal(0,0.01,200)
    y_data = np.square(x_data) + nouse
    x_data = np.reshape(x_data, (-1, 1))
    y_data = np.reshape(y_data, (-1, 1))

    #输入值占位置符
    x = tf.compat.v1.placeholder(tf.float32,shape=(200,1))
    y = tf.compat.v1.placeholder(tf.float32,shape=(200,1))

    #输入层权重
    weight_1 = tf.Variable(tf.random.normal((1,10)),dtype=tf.float32)
    b_1 = tf.Variable(tf.random.normal((1,10)), dtype=tf.float32)

    #输出层权重
    weight_2 = tf.Variable(tf.random.normal((10,1)),dtype=tf.float32)

    y_pred = tf.matmul(x, weight_1) + b_1
    y_pred = tf.nn.tanh(y_pred)
    y_pred = tf.matmul(y_pred, weight_2)
    y_pred = tf.nn.tanh(y_pred)

    loss = tf.reduce_mean(tf.square(y_pred - y))
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)
    optimizer = tf.compat.v1.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

    init = tf.compat.v1.initializers.global_variables()
    with tf.compat.v1.Session() as session:
        session.run(init)
        for step in range(200):
            session.run(optimizer, feed_dict={x:x_data, y:y_data})
            if step % 20 == 0:
                loss_var = session.run(loss, feed_dict={x:x_data, y:y_data})
                print("loss:{}".format(loss_var))

        y_p_data = session.run(y_pred, feed_dict={x:x_data})
    fig = plt.figure()

    plt.scatter(np.reshape(x_data, (-1)), np.reshape(y_data, (-1)))
    plt.plot(np.reshape(x_data, (-1)), np.reshape(y_p_data, (-1)), color='red')
    plt.show()


if __name__ == "__main__":
    simple_net_reg()