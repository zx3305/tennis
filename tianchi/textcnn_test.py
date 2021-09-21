import pandas as pd
import collections
import numpy as np
import pickle
from progressbar import *
import tensorflow.compat.v1 as tf
import gc
import time
import sys


class Config(object):
    def __init__(self):
        #词向量长度
        self.word_embedding_len = 128
        #训练文件目录
        self.train_file = '/Users/zero/Downloads/recommend/nlp/train_set.csv'
        #测试文件目录
        self.test_file = '/Users/zero/Downloads/recommend/nlp/test_a.csv'
        #输出目录
        self.out_path = './data'
        #输入文章长度
        self.sequence_length = 1000
        #正则化系数
        self.l2_reg_lambda = 0.001

class TextCnnNet(object):
    '''
    构建text-cnn网络
    '''
    def __init__(self, num_classes, vocabulary_len, embedding_vectors=[]):
        '''
        num_classes 分类数量
        vocabulary_len 词向量长度
        '''
        config = Config()
        with tf.name_scope("net_input"):
            self.input_x = tf.placeholder(tf.int32, [None, config.sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
            y = tf.one_hot(tf.cast(self.input_y, dtype=tf.int32), num_classes)
            self.drop_out = tf.placeholder(tf.float32, name="drop_out")
            self.lr = tf.Variable(0.002, dtype=tf.float32,trainable=False, name="lr")

            #定义l2正则
            l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            if len(embedding_vectors) > 0:
                #用外部词向量初始化
                embedding_w = tf.Variable(tf.cast(embedding_vectors, dtype=tf.float32),name="embedding_w")
            else:
                #随机初始化个词向量
                embedding_w = tf.Variable(tf.random.truncated_normal([vocabulary_len, config.word_embedding_len], stddev=0.1))
            embeading_lookup_origin = tf.nn.embedding_lookup(embedding_w, self.input_x)
            #扩展一个维度[batch_size, vocabulary_len,embedding_len, 1]
            embeading_lookup = tf.cast(tf.expand_dims(embeading_lookup_origin, -1), dtype=tf.float32)

        pool_outs = []
        with tf.name_scope("cnn"):
            #定义卷积核尺寸
            filter_sizes = [3,4,5]
            #卷积输出神经元数量
            filter_num = 128
            for i,filter_size in enumerate(filter_sizes):
                w = tf.Variable(tf.random.truncated_normal([filter_size, config.word_embedding_len, 1, filter_num], stddev=.1), name="cnn_w_"+str(i), dtype=tf.float32)
                b = tf.Variable(tf.zeros([filter_num]), dtype=tf.float32, name="cnn_b_"+str(i))
                conv = tf.nn.conv2d(embeading_lookup, w,strides=[1, 1, 1, 1] ,padding="VALID", name="cnn_conv_"+str(i))
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                #最大池化
                h_size = config.sequence_length - filter_size + 1
                max_pool = tf.nn.max_pool(h, [1, h_size, 1, 1], padding="VALID", strides=[1, 1, 1, 1])
                pool_outs.append(max_pool)
        with tf.name_scope("concat"):
            #池化拼接
            flat = tf.concat(pool_outs, 3)
            #把池化层变成一维向量
            flat_x = tf.reshape(flat,[-1, 3*filter_num])
            #drop_out
            flat_x_drop_out = tf.nn.dropout(flat_x, self.drop_out)
        with tf.name_scope("output"):
            out_w = tf.Variable(tf.random.truncated_normal([3*filter_num, num_classes], stddev=.1))
            out_b = tf.Variable(tf.zeros([num_classes]), name="out_b")
            l2_loss += tf.nn.l2_loss(out_w)
            l2_loss += tf.nn.l2_loss(out_b)
            self.output = tf.nn.softmax(tf.nn.xw_plus_b(flat_x_drop_out, out_w, out_b))
        with tf.name_scope("train"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.output)
            self.loss = tf.reduce_mean(losses, name="loss") + config.l2_reg_lambda*l2_loss
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(self.output, 1)), dtype=tf.float32),name="accuracy")
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, name="optimizer")        


class DataSet(object):
    def __init__(self):
        self.config = Config()

    def data_to_tfrecode(self):
        df = pd.read_csv(self.config.test_file, sep='\t')
        with open(self.config.out_path+"/vocab.pkl", "rb") as f:
            d = pickle.load(f)
        #词典
        vocab_dict = list(d['idx2word'].values())

        tf_write = tf.python_io.TFRecordWriter(self.config.out_path+"/test_data.tfrecode")
        pbar = ProgressBar().start()
        for i,word in enumerate(list(df['text'])):
            word_arr = [str(d['word2idx'][str(v)]) for v in word.split() if str(v) in vocab_dict]
            if len(word_arr) >= self.config.sequence_length:
                word_arr = word_arr[:self.config.sequence_length]
            else:
                word_arr = word_arr + [str(d['word2idx']['PAD'])]*(self.config.sequence_length - len(word_arr))
            x_str = ",".join(word_arr)
            x_str = x_str.encode("utf-8")
            example = tf.train.Example(features = tf.train.Features(feature={
                "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_str]))
            }))
            tf_write.write(example.SerializeToString())
            pbar.update((i / (len(df['text']) - 1)) * 100)
        pbar.finish()

    def get_dataset(self):
        dataset = tf.data.TFRecordDataset(self.config.out_path+"/test_data.tfrecode")
        def parse_exmp(serial_exmp):
            feats = tf.parse_single_example(serial_exmp, features={
                "text": tf.FixedLenFeature([], tf.string)
            })
            text = feats['text']
            return text
        return dataset.map(parse_exmp)

    def cluster_data(self):
        df = pd.read_csv(self.config.test_file, sep='\t')
        with open(self.config.out_path+"/vocab.pkl", "rb") as f:
            d = pickle.load(f)
        #词典
        vocab_dict = list(d['idx2word'].values())
        pbar = ProgressBar().start()
        file_num = 0
        cache_data = []
        for i,word in enumerate(list(df['text'])):
            word_arr = [str(d['word2idx'][str(v)]) for v in word.split() if str(v) in vocab_dict]
            if len(word_arr) >= self.config.sequence_length:
                word_arr = word_arr[:self.config.sequence_length]
            else:
                word_arr = word_arr + [str(d['word2idx']['PAD'])]*(self.config.sequence_length - len(word_arr))
            x_str = ",".join(word_arr)
            cache_data.append(x_str)
            if i >0 and i%2000 == 0:
                with open("./test/test_"+str(file_num)+".text", "w+") as f:
                    f.write("\n".join(cache_data))
                file_num += 1
                cache_data = []
            pbar.update((i / (len(df['text']) - 1)) * 100)
        pbar.finish()
        if len(cache_data) > 0:
            with open("./test/test_"+str(file_num)+".text", "w+") as f:
                f.write("\n".join(cache_data))


start = 0
end = 25
if len(sys.argv) > 2:
    start = int(sys.argv[1])
    end = int(sys.argv[2])


dset = DataSet()
textcnn = TextCnnNet(num_classes=14, vocabulary_len=5950, embedding_vectors=[])
config = Config()
saver = tf.train.Saver()

with tf.Session() as sess:
    #检查最新加载点
    ckpt = tf.train.get_checkpoint_state('./data')
    #恢复图结构和数据
    saver.restore(sess, ckpt.model_checkpoint_path)

    output = textcnn.output
    input_x = textcnn.input_x
    drop_out = textcnn.drop_out

    for file_i in range(start, end):
        with open("./test/test_"+str(file_i)+".text", "r") as f:
            content = f.read()
        data = content.split("\n")
        retArray = []
        for i,text in enumerate(data):
            train_text = [ [int(vv) for vv in v.split(",")] for v in [text]]
            train_text = np.reshape(np.array(train_text), [-1, config.sequence_length])
            output_v = sess.run(output, feed_dict={input_x:train_text, drop_out:1})
            ret = sess.run(tf.argmax(output_v,1))
            retArray.append(str(ret[0]))
            print("ret:{}".format(str(i)+":"+str(ret[0])))
        with open("./ret/ret_"+str(file_i)+".txt", "w+") as f:
            f.write(",".join(retArray))
        retArray = []

