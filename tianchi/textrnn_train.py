import pandas as pd
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import collections
import numpy as np
import pickle
from progressbar import *
import tensorflow.compat.v1 as tf
import gc
import time

class Config(object):
    def __init__(self):
        #句子长度
        self.sequence_length = 1000
        #词向量长度
        self.word_embedding_len = 128
        self.vocabulary_len = 6000
        #分类
        self.num_classes = 14
        #隐藏层神经元
        self.hidden_dim = 128
        #rnn神经元类型
        self.rnn_type = "lstm"
        # 隐藏层层数
        self.num_layers= 2
        #输出目录
        self.out_path = './data'

class TextRnnNet(object):
    def __init__(self, config):
        with tf.name_scope("net_input"):
            self.input_x = tf.placeholder(tf.int32, [None, config.sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.int32, [None])
            y = tf.one_hot(tf.cast(self.input_y, dtype=tf.int32), config.num_classes)
            self.drop_out = tf.placeholder(tf.float32, name="drop_out")
            self.lr = tf.Variable(0.002, dtype=tf.float32,trainable=False, name="lr")

        with tf.name_scope("embedding"):
             #随机初始化个词向量
            embedding_w = tf.Variable(tf.random.truncated_normal([config.vocabulary_len, config.word_embedding_len], stddev=0.1))
            embeading_lookup = tf.nn.embedding_lookup(embedding_w, self.input_x)

        with tf.name_scope("rnn_net"):
            cells = [self.cell(config) for _ in range(config.num_layers)]
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            outputs_,state = tf.nn.dynamic_rnn(cells, inputs=embeading_lookup, dtype=tf.float32)

            #[batch_size, 1 ,hidden_dim]
            final_output = outputs_[:,-1,:]
            final_output = tf.reshape(final_output, [-1, config.hidden_dim])
            #全链接层
            with tf.name_scope("fc"):
                w1 = tf.Variable(tf.random.truncated_normal([config.hidden_dim, config.hidden_dim]), dtype=tf.float32)
                b1 = tf.Variable(tf.zeros([config.hidden_dim]), dtype=tf.float32)
                fc1 = tf.matmul(final_output, w1) + b1
                fc1 = tf.nn.dropout(fc1, self.drop_out)
                fc1 = tf.nn.relu(fc1)
            #输出层
            with tf.name_scope("output"):
                w2 = tf.Variable(tf.random.truncated_normal([config.hidden_dim, config.num_classes]), dtype=tf.float32)
                b2 = tf.Variable(tf.zeros([config.num_classes]), dtype=tf.float32)
                fc2 = tf.matmul(fc1, w2) + b2
                self.output = tf.nn.softmax(fc2)

        with tf.name_scope("train"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.output))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(self.output, 1)), dtype=tf.float32),name="accuracy")

    def cell(self, config):
        if config.rnn_type == 'lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(config.hidden_dim)
        else:
            return tf.nn.rnn_cell.GRUCell(config.hidden_dim)

class DateSet(object):
    '''
    数据处理
    '''
    def __init__(self):
        self.config = Config()

    def create_vocab(self):
        '''
        把训练数据转换成tfrecode格式数据
        '''
        df = pd.read_csv(self.config.train_file, sep='\t')
        #词典处理
        allwords = [word for lines in list(df['text']) for word in lines.split()]
        wordCount = collections.Counter(allwords)
        wordCount = sorted(wordCount.items(), key=lambda x:x[1], reverse=True)
        #过滤出现次数小于5的词，过滤出现次数top5的词
        top5 = [x[0] for x in wordCount[:5]]
        wordCount = [x for x in wordCount if x[1]>5 and x[0] not in top5]

        #定义词典
        vocab = []
        vocab.append('PAD')
        vocab.append('UNK')
        for x in wordCount:
            vocab.append(x[0])
        word2idx = dict(zip(vocab, list(range(len(vocab)))))
        idx2word = dict(zip(list(range(len(vocab))),vocab))
        d = {"word2idx":word2idx, "idx2word":idx2word}
        print(len(vocab))
        with open(self.config.out_path+"/vocab.pkl", "wb+") as f:
            pickle.dump(d, f)

    def data_to_tfrecode(self, type="train"):
        df = pd.read_csv(self.config.train_file, sep='\t')
        if type == "train":
            df = df[100000:190000]
        else:
            df = df[190000:]

        with open(self.config.out_path+"/vocab.pkl", "rb") as f:
            d = pickle.load(f)
        #词典
        vocab_dict = list(d['idx2word'].values())

        tf_write = tf.python_io.TFRecordWriter(self.config.out_path+"/"+type+"2_data.tfrecode")
        pbar = ProgressBar().start()
        for i,word in enumerate(list(df['text'])):
            word_arr = [str(d['word2idx'][str(v)]) for v in word.split() if str(v) in vocab_dict]
            if len(word_arr) >= self.config.sequence_length:
                word_arr = word_arr[:self.config.sequence_length]
            else:
                word_arr = word_arr + [str(d['word2idx']['PAD'])]*(self.config.sequence_length - len(word_arr))
            x_str = ",".join(word_arr)
            x_str = x_str.encode("utf-8")
            label = list(df['label'])[i]
            example = tf.train.Example(features = tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_str]))
            }))
            tf_write.write(example.SerializeToString())
            pbar.update((i / (len(df['text']) - 1)) * 100)
        pbar.finish()

    def get_dataset(self, type="train"):
        dataset = tf.data.TFRecordDataset(self.config.out_path+"/"+type+"_data.tfrecode")
        def parse_exmp(serial_exmp):
            feats = tf.parse_single_example(serial_exmp, features={
                "label": tf.FixedLenFeature([], tf.int64),
                "text": tf.FixedLenFeature([], tf.string)
            })
            label = tf.cast(feats['label'], dtype=tf.int8)
            text = feats['text']
            return label,text
        return dataset.map(parse_exmp)

config = Config()
trn = TextRnnNet(config)

dset = DateSet()
total_nums = 100000
batch_size = 512
batch_num = total_nums // batch_size
epochs = 12

with tf.Session() as sess:
    dset_next = dset.get_dataset().shuffle(total_nums).repeat(epochs).batch(batch_size).make_one_shot_iterator().get_next()
    sess.run(tf.global_variables_initializer())

    input_x = trn.input_x
    input_y = trn.input_y
    accuracy = trn.accuracy
    loss = trn.loss
    lr = trn.lr
    drop_out = trn.drop_out
    optimizer = trn.optimizer
    for epoch in range(epochs):
        for b_i in range(batch_num):
            label,text = sess.run(dset_next)
            train_text = [ [int(i) for i in v.decode("utf-8").split(",")] for v in text]
            train_text = np.reshape(np.array(train_text), [-1, config.sequence_length])
            sess.run(optimizer, feed_dict={input_x:train_text, input_y:label, drop_out:0.8})
            if b_i % 100 == 0:
                accuracy_v,loss_v = sess.run([accuracy,loss], feed_dict={input_x:train_text, input_y:label, drop_out:1.0})
                print("epoch:{},accuracy:{},loss:{}".format(epoch, accuracy_v, loss_v))                    




