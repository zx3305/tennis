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


'''
数据集：新闻文本推荐，天池的学习赛
数据格式csv：label：1-14 表示新闻14哥分类，text：57 44 66 56 ... 用数字替代了文字，所以无法识别停止词、标点符号等。
思路：
(一): 数据处理步骤，读取了训练集和测试集合所有的数据，一个文章一行，然后使用gensim.models.word2vec词向量化。
(二)：遍历所有所有的训练集数据，计算出字典，其中文字频率top5移除(疑似标点符号等)、小于出现5次的文字移除。
(三)：把训练数据的文字转换成字典对应的编号，每篇文章保留1000个文字，不足1000的用0填充，并把数据格式转为tfrecode格式
(四): 使用textcnn 网络训练模型，其中词向量维度为128。
'''

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


class DateSet(object):
    '''
    数据处理
    '''
    def __init__(self):
        self.config = Config()

    def genim_word2vec(self):
        '''
        word2vec方法
        '''
        df_train = pd.read_csv(self.config.train_file, sep='\t')
        f = open(self.config.out_path+"/text.txt", "a+")
        for text in list(df_train['text']):
            f.write(text + "\n")
        del df_train
        gc.collect()
        df_test = pd.read_csv(self.config.test_file, sep='\t')
        for text in list(df_test['text']):
            f.write(text + "\n")
        f.close()

        lineSen = LineSentence(self.config.out_path+"/text.txt")
        model = Word2Vec(lineSen, size=self.config.word_embedding_len, window=5, min_count=1, workers=4)
        model.save(self.config.out_path+"/word2vec.model")
        print(np.shape(model.wv.vectors))

    def get_embedding(self):
        with open(self.config.out_path+"/vocab.pkl", "rb") as f:
            d = pickle.load(f)
        model = Word2Vec.load(self.config.out_path+"/word2vec.model")
        words = list(d['word2idx'].keys())
        embedding_vectors = []
        for word in words:
            try:
                vector = model.wv[word]
                embedding_vectors.append(vector)
            except:
                print(word + "不存在于词向量中")
        return np.array(embedding_vectors)


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

dset = DateSet()
# dset.data_to_tfrecode()
# exit(0)
# embedding_vectors = dset.get_embedding()

textcnn = TextCnnNet(num_classes=14, vocabulary_len=5950, embedding_vectors=[])
config = Config()
total_nums = 100000
total_nums_2 = 90000
batch_size = 512
batch_num = total_nums // batch_size
batch_num_2 = total_nums_2 // batch_size
epochs = 12
saver = tf.train.Saver()
start_time = time.clock()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #检查最新加载点
    ckpt = tf.train.get_checkpoint_state('./data')
    #恢复图结构和数据
    saver.restore(sess, ckpt.model_checkpoint_path)

    dset_next = dset.get_dataset().shuffle(total_nums).repeat(epochs).batch(batch_size).make_one_shot_iterator().get_next()
    dset_next2 = dset.get_dataset("train2").shuffle(total_nums_2).repeat(epochs).batch(batch_size).make_one_shot_iterator().get_next()
    test_dset = dset.get_dataset("test").batch(batch_size).make_one_shot_iterator().get_next()
    input_x = textcnn.input_x
    input_y = textcnn.input_y
    accuracy = textcnn.accuracy
    loss = textcnn.loss
    lr = textcnn.lr
    drop_out = textcnn.drop_out
    optimizer = textcnn.optimizer

    for epoch in range(epochs):
        if epoch>0 and epoch % 3 == 0:
            sess.run(tf.assign(lr, lr/1.5))
        if epoch <= 9:
            continue
        for b_i in range(batch_num):
            label,text = sess.run(dset_next)
            train_text = [ [int(i) for i in v.decode("utf-8").split(",")] for v in text]
            train_text = np.reshape(np.array(train_text), [-1, config.sequence_length])
            sess.run(optimizer, feed_dict={input_x:train_text, input_y:label, drop_out:0.8})
            if b_i % 100 == 0:
                accuracy_v,loss_v = sess.run([accuracy,loss], feed_dict={input_x:train_text, input_y:label, drop_out:1.0})
                print("epoch:{},accuracy:{},loss:{}".format(epoch, accuracy_v, loss_v))                
        for a_i in range(batch_num_2):
            label,text = sess.run(dset_next2)
            train_text = [ [int(i) for i in v.decode("utf-8").split(",")] for v in text]
            train_text = np.reshape(np.array(train_text), [-1, config.sequence_length])
            sess.run(optimizer, feed_dict={input_x:train_text, input_y:label, drop_out:0.8})
            if a_i % 100 == 0:
                accuracy_v,loss_v = sess.run([accuracy,loss], feed_dict={input_x:train_text, input_y:label, drop_out:1.0})
                print("epoch:{},accuracy:{},loss:{}".format(epoch, accuracy_v, loss_v))

        test_label,text_text = sess.run(test_dset)
        text_text = [ [int(i) for i in v.decode("utf-8").split(",")] for v in text_text]
        text_text = np.reshape(np.array(text_text), [-1, config.sequence_length])
        accuracy_v,loss_v = sess.run([accuracy,loss], feed_dict={input_x:text_text, input_y:test_label, drop_out:1.0})
        print("")
        print("耗时:{}秒".format(time.clock() - start_time))
        start_time = time.clock()
        print("epoch:{},accuracy:{},loss:{}".format(epoch, accuracy_v, loss_v))
        print("")
        saver.save(sess, "./data/textcnn.mdl", global_step=epoch)



