import numpy as np
from sklearn.naive_bayes import MultinomialNB

def getDataSet():
    """
    加载训练数据, postingList是所有的训练集, 每一个列表代表一条言论, 一共有8条言论 classVec代表每一条言论的类别,
     0是正常, 1是有侮辱性 返回 言论和类别
    :return:
    """
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0,1,0,1,0,1]
    return postingList,labels

def createVocabList(dataSet):
    """
    创建词汇表, 就是把这个文档中所有的单词不重复的放在一个列表里面
    :param dataSet:
    :return:
    """
    vocabSet = set([])
    for data in dataSet:
        vocabSet = vocabSet | set(data) #集合求或运算
    return list(vocabSet)

def vectorize(vocabSet,dataSet):
    """
    制作词向量矩阵
    将每一个文档转换为词向量, 然后放入矩阵中
    :param vocabSet:
    :param dataSet:
    :return:
    """
    vocab = [0] * len(vocabSet)
    for data in dataSet:
        vocab[vocabSet.index(data)] = 1
    return vocab

def trainN(X_train,y_train):
    """
    制作贝叶斯分类器
    :param X_train:
    :param y_train:
    :return:
    """
    num = len(X_train)   #有多少记录
    numvocab = len(X_train[0]) #词向量的大小
    p0Num = np.ones(numvocab) #统计非侮辱类的相关单词频数 加入了拉普拉斯平滑
    p1Num = np.ones(numvocab) #统计侮辱类的相关单词频数
    p0Sum = 2
    p1Sum = 2
    pA = sum(y_train) / num                   #先验概率
    for i in range(num):
        if y_train[i]==0:   #统计属于非侮辱类的条件概率所需的数据
            p0Sum += 1
            p0Num += X_train[i]
        else:               #统计属于侮辱类的条件概率所需的数据
            p1Sum += 1
            p1Num += X_train[i]

    # 为了防止下溢出，计算条件概率的对数
    p0 = np.log(p0Num / p0Sum)      #频数除以总数 得到概率
    p1 = np.log(p1Num / p1Sum)
    return p0,p1,pA


def classify(testMat,p0,p1,pA):
    """
    进行分类
    :param testMat: 
    :param p0: 
    :param p1: 
    :param pA: 
    :return: 
    """
    p0Score = sum(testMat * p0) +pA
    p1Score = sum(testMat * p1) +(1-pA)     #在计算概率时已经取了对数，直接求和比较比较大小
    if p0Score > p1Score:
        return 0
    else:
        return 1

if __name__=='__main__':
    dataSet,label = getDataSet()
    vocabSet = createVocabList(dataSet)
    trainMat = []
    for elem in dataSet:
        trainMat.append(vectorize(vocabSet,elem))
    p0,p1,pA = trainN(trainMat,label)
    
    test1= ['love', 'my', 'dalmation']
    test2= ['stupid', 'garbage','love']
    test1_vocab = np.array(vectorize(vocabSet,test1))
    test2_vocab = np.array(vectorize(vocabSet,test2))
    result1 = classify(test1_vocab,p0,p1,pA)
    result2 = classify(test2_vocab,p0,p1,pA)
    print([result1, result2])

    #多项式朴素贝叶斯算法
    mnb = MultinomialNB(alpha=1.0)    #1拉普拉斯平滑
    mnb.fit(np.array(trainMat), np.array(label).reshape(6,))
    y_pred_mnb = mnb.predict(np.array([test1_vocab, test2_vocab]))
    print(y_pred_mnb)
