'''
假设有一批金融机构借款者的信息，根据这些信息金融机构已经进行了信贷评级（credit）1、2、3
特征变了有5个，income(借款者的月收入，单位：元)、house(借款者拥有的房产数，单位：处)、
points(信用积分，单位：分)、default(之前的不良违约次数，单位：次)
'''
#贝叶斯分类算法

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def draw(df):
    a1 = df[df['credit'] == 1]
    a2 = df[df['credit'] == 2]
    a3 = df[df['credit'] == 3]
    fig,ax = plt.subplots(figsize=(8,5))
    ax.scatter(a1['income'], a1['points'], s=30, c='b', marker='o')
    ax.scatter(a2['income'], a2['points'], s=30, c='r', marker='x')
    ax.scatter(a3['income'], a3['points'], s=30, c='g', marker='^')
    ax.legend()
    ax.set_xlabel('income')
    ax.set_ylabel('points')
    plt.show()

def sort(df):
    x = df.iloc[:, 1:5]
    y = df['credit']
    x = np.array(x.values)
    y = np.array(y.values)
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=.25, random_state=1)

    #高斯朴素贝叶斯算法
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    # print(gnb.class_prior_)     #先验证概率
    # print(gnb.class_count_)     #训练样本数
    # print(gnb.theta_)           #特征值上的均值
    # print(gnb.sigma_)           #特征上的方差
    #高斯朴素贝叶斯算法
    y_pred = gnb.predict(x_test)
    print(accuracy_score(y_test, y_pred))  #计算准确率
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))


    #多项式朴素贝叶斯算法
    mnb = MultinomialNB(alpha=1.0)    #1拉普拉斯平滑
    mnb.fit(x_train, y_train)
    y_pred_mnb = mnb.predict(x_test)
    print(accuracy_score(y_test, y_pred_mnb))

    #伯努利朴素贝叶斯算法
    bnb = BernoulliNB(alpha=1.0, binarize=2.0, fit_prior=True)
    bnb.fit(x_train, y_train)
    y_pred_bnb = bnb.predict(x_test)
    print(accuracy_score(y_test, y_pred_bnb))

if __name__ == '__main__':
    df = pd.read_csv('./tain.csv')
    sort(df)
    # draw(df)


