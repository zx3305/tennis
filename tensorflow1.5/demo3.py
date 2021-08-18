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
#tensorboard
#https://github.com/tensorflow/tensorboard/blob/master/docs/r1/overview.md

batch_num = 200

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32,shape=[None, 784], name='input_x')
    y = tf.placeholder(dtype=tf.float32,shape=[None, 10], name='input_y')
    keep_prob = tf.placeholder(dtype=tf.float32, name = 'keep_prob')

with tf.name_scope('layer_1'):
    weight_1 = tf.Variable(tf.random.truncated_normal([784,500]), dtype=tf.float32, name='weight_1')
    tf.summary.scalar('weight_1', tf.reduce_mean(weight_1))
    tf.summary.histogram("weight_1_h", weight_1)
    b_1 = tf.Variable(tf.zeros([1, 500]), dtype=tf.float32, name='b_1')
    tf.summary.scalar('b_1', tf.reduce_mean(b_1))
    l1 = tf.nn.tanh(tf.matmul(x, weight_1) + b_1)
    l1 = tf.nn.dropout(l1, keep_prob=keep_prob)

with tf.name_scope('layer_2'):
    weight_2 = tf.Variable(tf.random.truncated_normal([500,100]), dtype=tf.float32, name='weight_2')
    # tf.summary.scalar('weight_2', weight_2)
    b_2 = tf.Variable(tf.zeros([1, 100]), dtype=tf.float32, name='b_2')
    # tf.summary.scalar('b_2', b_2)
    l2 = tf.nn.tanh(tf.matmul(l1, weight_2) + b_2)
    l2 = tf.nn.dropout(l2, keep_prob=keep_prob)

with tf.name_scope('layer_3'):
    weight_3 = tf.Variable(tf.random.truncated_normal([100,10]), dtype=tf.float32, name='weight_3')
    # tf.summary.scalar('weight_3', weight_3)
    b_3 = tf.Variable(tf.zeros([1, 10]), dtype=tf.float32, name='b_3')
    # tf.summary.scalar('b_3', b_3)
    l3 = tf.matmul(l2, weight_3) + b_3

#模型输出
with tf.name_scope('output'):
    y_pred = tf.nn.softmax(l3, name='softmax')
    #正确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1)),dtype=tf.float32), name='accuracy')

with tf.name_scope('train'):
    #损失函数
    lr = tf.Variable(0.01, dtype=tf.float32,trainable=False)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred), name='loss')
    tf.summary.scalar('loss', loss)
    #优化器
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

l = mnist.train.num_examples // batch_num
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lr, 0.01*((51-epoch)/40)))
        write = tf.summary.FileWriter("./tensorboard", sess.graph)
        for i in range(l-1):
            x_data,y_data=mnist.train.next_batch(batch_num)
            sess.run(optimizer, feed_dict={x:x_data, y:y_data, keep_prob:.7})
            merge_data = sess.run(merged, feed_dict={x:x_data, y:y_data, keep_prob:.7})
        if epoch % 5 == 0:
            accuracy_t = sess.run(accuracy, feed_dict={x:x_data, y:y_data, keep_prob:1.0})
            test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images[:1000], y:mnist.test.labels[:1000], keep_prob:1.0})
            print("accuracy:{},test_accuracy:{}".format(accuracy_t,test_accuracy))
        write.add_summary(merge_data, global_step=epoch)






