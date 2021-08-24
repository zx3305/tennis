import tensorflow.compat.v1 as tf
import numpy as np

import warnings
warnings.filterwarnings("ignore")

'''
使用RNN神经网络,预测双色球
数据来源网址：http://zst.aicai.com/ssq/openInfo/
'''

def getData():
    '''
    解析双色球数据
    输出：(-1, 36) 
    '''
    with open('./双色球数据/ssqtxt_result.txt', 'r') as f:
        ret = np.array([])
        line = f.readline()
        while line:
            arr = line.split("\t")
            if len(arr) == 4 and arr[3].find(",") > -1:
                ssq_num = arr[3].replace("\\", ",")
                ssq_num = ssq_num.replace("\n", "")
                ssq_num = ssq_num.split(",")
                ssq_num = np.array(ssq_num).astype(int)/36
                if len(ssq_num) == 7 :
                    ssq_num = ssq_num.reshape((1,7))
                    if len(ret) == 0:
                        ret = ssq_num
                    else:
                        ret = np.concatenate((ret,ssq_num))
                else:
                    pass
                    # print(line)

            line = f.readline()
    return ret[::-1]


#网络构建
num_unit = 64
max_time = 10

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 7])
        y = tf.placeholder(tf.float32, [None, 7])
        x_input = tf.reshape(x, [-1, max_time, 7])

    with tf.name_scope("rnn"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_unit)
        output,state = tf.nn.dynamic_rnn(lstm_cell, x_input, dtype=tf.float32)
        weight = tf.Variable(tf.random.truncated_normal([num_unit, 7], stddev=0.1), dtype=tf.float32)
        b = tf.Variable(tf.random.truncated_normal([7], stddev=0.1), dtype=tf.float32)
        pred = tf.matmul(state[1], weight) + b

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(y - pred))

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(0.0003).minimize(loss)
        y_ = tf.cast(y*36, dtype=tf.int8)
        pred_ = tf.cast(pred*36, dtype=tf.int8)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_, pred_),dtype=tf.float32))

    data = getData()
    num = np.shape(data)[0]
    batch_num = num // (max_time + 1)
    #模型保存
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1):
        for i in range(num - max_time - 1):
            s = i 
            e = i + max_time
            x_data = data[s:e]
            y_data = data[e:e+1]
            sess.run(optimizer, feed_dict={x:x_data, y:y_data})
        accuracy_data = sess.run(accuracy, feed_dict={x:x_data, y:y_data})
        loss_data = sess.run(loss, feed_dict={x:x_data, y:y_data})
        print("accuracy:{},loss:{}".format(accuracy_data, loss_data))

    print(sess.run(pred_, feed_dict={x:x_data}))
    print(sess.run(y_, feed_dict={y:y_data}))
    #模型保存
    saver.save(sess, "./saver/ssq.pkl")
    #模型保存为二进制rb文件
    print(graph.get_name_scope())
    # gd = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ["train"])
    # with tf.gfile.Gfile("./saver/ssq.pb", "wb") as f:
    #     f.write(gd.SerializeToString())





