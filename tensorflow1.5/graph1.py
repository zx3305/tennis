import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import warnings
warnings.filterwarnings("ignore")

'''
模型下载地址
链接: https://pan.baidu.com/s/1kqtKnsfA7snKqMNSRz86ZQ 
提取码: aj1b 

学习地址：https://www.cnblogs.com/hellcat/p/6925757.html
'''

def load_graph():
    '''
    载入数据和图
    '''
    with tf.Session() as sess:
        with tf.gfile.FastGFile("/Users/zero/Downloads/inception_model/classify_image_graph_def.pb", "rb") as f:
            #新建GrapDef文件，用于载入模型中的图
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            #载入到当前默认图中
            tf.graph_util.import_graph_def(graph_def)
        #使用tensorborad查看结构
        tf.summary.FileWriter("./tensorboard", sess.graph)


def load_graph2():
    '''
    载入数据和图
    '''
    #检查最新加载点
    ckpt = tf.train.get_checkpoint_state('./saver')
    #载入图结构，保存在.meta文件中
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+".meta")
    with tf.Session() as sess:
        #恢复图结构和数据
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(tf.get_default_graph().get_operations())
        # tf.summary.FileWriter("./tensorboard", tf.get_default_graph())

        #sess.run(tf.get_default_graph().get_tensor_by_name()) 取的tensor做运算


if __name__ == '__main__':
    load_graph2()

