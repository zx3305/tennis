import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import warnings
warnings.filterwarnings("ignore")

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch = 200
batch_num = mnist.train.num_examples // batch

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y = tf.placeholder(tf.float32, [None, 10], name="y")
    x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")
    drop_out = tf.placeholder(tf.float32)

#卷积后的图像大小计算 new_height=(old_height−F+2×P)/S +1 ; F核大小、P填充大小、S为步长
#池化后的大小 H = (H - F) / S + 1
#卷积层1
with tf.name_scope("conv1"):
    conv_1_w = tf.Variable(tf.random.truncated_normal([3,3,1,32], stddev=0.1),dtype=tf.float32, name='conv_1_w')
    conv_1_b = tf.Variable(tf.zeros([32]), dtype=tf.float32, name='conv_1_b')
    conv_1 = tf.nn.relu(tf.nn.conv2d(x_image, conv_1_w, padding="SAME") + conv_1_b, name="conv_1")
    #池化后的图片大小(14*14)：(28-2) / 2 + 1 = 14
    conv_1_pool = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="conv_1_pool")

#卷积层2
with tf.name_scope("conv2"):
    conv_2_w = tf.Variable(tf.random.truncated_normal([3,3,32,64],stddev=0.1),dtype=tf.float32, name='conv_2_w')
    conv_2_b = tf.Variable(tf.zeros([64]), dtype=tf.float32, name='conv_2_b')
    conv_2 = tf.nn.relu(tf.nn.conv2d(conv_1_pool, conv_2_w, padding="SAME") + conv_2_b, name="conv_2")
    #池化后的图片大小(7*7)：(14-2) / 2 + 1 = 7
    conv_2_pool = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="conv_2_pool")

#全连接层
with tf.name_scope("full_c"):
    #展开
    full_input = tf.reshape(conv_2_pool, [-1, 7*7*64])
    full_c_w = tf.Variable(tf.random.truncated_normal([7*7*64, 256],stddev=0.1), dtype=tf.float32, name='full_c_w')
    full_c_b = tf.Variable(tf.zeros([256]), dtype=tf.float32, name='full_c_b')
    full_c = tf.nn.relu(tf.matmul(full_input, full_c_w) + full_c_b, name="full_c")
    full_c = tf.nn.dropout(full_c, keep_prob=drop_out)

#输出层
with tf.name_scope("output"):
    out_w = tf.Variable(tf.random.truncated_normal([256, 10],stddev=0.1), dtype=tf.float32, name='out_w')
    out_b = tf.Variable(tf.zeros([10]), dtype=tf.float32, name="out_b")
    out = tf.nn.softmax(tf.matmul(full_c, out_w) + out_b)

#训练
with tf.name_scope("train"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out))
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

#准确率
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)),dtype=tf.float32))
    tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    train_file = tf.summary.FileWriter("./tensorboard/train", session.graph)
    test_file =tf.summary.FileWriter("./tensorboard/test", session.graph)
    for epoch in range(10):
        for _ in range(batch_num):
            x_data,y_data = mnist.train.next_batch(batch)
            session.run(optimizer, feed_dict={x:x_data, y:y_data, drop_out:0.7})
            summary = session.run(merged, feed_dict={x:x_data, y:y_data, drop_out:1.0})

        accuracy_data = session.run(accuracy, feed_dict={x:mnist.test.images[:1000], y:mnist.test.labels[:1000],drop_out:1.0})
        test_summary = session.run(merged, feed_dict={x:mnist.test.images[:1000], y:mnist.test.labels[:1000],drop_out:1.0})
        print("accuracy:{}".format(accuracy_data))
        train_file.add_summary(summary, global_step=epoch)
        test_file.add_summary(test_summary, global_step=epoch)









