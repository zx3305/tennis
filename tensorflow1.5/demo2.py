import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import warnings
warnings.filterwarnings("ignore")

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#训练数据集(55000, 784), 55000代表图片数量, 28*28=784表示图片像素
# print(np.shape(mnist.train.images))
#标签(55000,10), one-hot编码
# print(np.shape(mnist.train.labels))

batch_num = 200

x = tf.placeholder(dtype=tf.float32,shape=[None, 784])
y = tf.placeholder(dtype=tf.float32,shape=[None, 10])
keep_prob = tf.placeholder(dtype=tf.float32)

weight_1 = tf.Variable(tf.random.normal([784,500]), dtype=tf.float32)
b_1 = tf.Variable(tf.zeros([1, 500]), dtype=tf.float32)
l1 = tf.nn.tanh(tf.matmul(x, weight_1) + b_1)
l1 = tf.nn.dropout(l1, keep_prob=keep_prob)

weight_2 = tf.Variable(tf.random.normal([500,100]), dtype=tf.float32)
b_2 = tf.Variable(tf.zeros([1, 100]), dtype=tf.float32)
l2 = tf.nn.tanh(tf.matmul(l1, weight_2) + b_2)
l2 = tf.nn.dropout(l2, keep_prob=keep_prob)


weight_3 = tf.Variable(tf.random.normal([100,10]), dtype=tf.float32)
b_3 = tf.Variable(tf.zeros([1, 10]), dtype=tf.float32)
l3 = tf.matmul(l2, weight_3) + b_3
# l2 = tf.nn.dropout(l1, keep_prob=keep_prob)

#模型输出
y_pred = tf.nn.softmax(l3)

#损失函数
# loss = tf.reduce_mean(tf.square(y - y_pred))
# loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))

#优化器
optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#正确率
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1)),dtype=tf.float32))

init = tf.global_variables_initializer()

l = mnist.train.num_examples // batch_num
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(30):
        for i in range(l-1):
            x_data,y_data=mnist.train.next_batch(batch_num)
            sess.run(optimizer, feed_dict={x:x_data, y:y_data, keep_prob:.7})
        if epoch % 5 == 0:
            accuracy_t = sess.run(accuracy, feed_dict={x:x_data, y:y_data, keep_prob:1.0})
            test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images[:1000], y:mnist.test.labels[:1000], keep_prob:1.0})
            print("accuracy:{},test_accuracy:{}".format(accuracy_t,test_accuracy))






