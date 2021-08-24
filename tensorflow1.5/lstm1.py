import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import warnings
warnings.filterwarnings("ignore")

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
手写数字识别的lstm网络
'''
max_time = 28 #一批数据序列最大长度
num_unit = 64 #隐层单元长度
batch = 100
batch_num = mnist.train.num_examples // batch

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y = tf.placeholder(tf.float32, [None, 10], name="y")
    x_image = tf.reshape(x, [-1, max_time, 28])

with tf.name_scope("rnn"):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_unit)
    outputs,state = tf.nn.dynamic_rnn(lstm_cell, x_image, dtype=tf.float32)
    weight = tf.Variable(tf.random.truncated_normal([num_unit, 10], stddev=0.1), dtype=tf.float32)
    b = tf.Variable(tf.random.truncated_normal([10], stddev=0.1), dtype=tf.float32)
    pred = tf.nn.softmax(tf.matmul(state[1], weight) + b)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)), dtype=tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter("./tensorboard", sess.graph)
    for epoch in range(10):
        for _ in range(batch_num):
            x_data, y_data = mnist.train.next_batch(batch)
            sess.run(optimizer, feed_dict={x:x_data, y:y_data})
        accuracy_data = sess.run(accuracy, feed_dict={x:mnist.test.images[:1000], y:mnist.test.labels[:1000]})
        print("accuracy:{}".format(accuracy_data))
    saver.save(sess, "./saver/lstm1.pkl")

with tf.Session() as sess:
    saver.restore(sess, "./saver/lstm1.pkl")
    accuracy_data = sess.run(accuracy, feed_dict={x:mnist.test.images[1000:2000], y:mnist.test.labels[1000:2000]})
    print("restore_accuracy:{}".format(accuracy_data))
