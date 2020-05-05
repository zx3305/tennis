#还是以平安银行(000001) 2019-01-01 的股票数据来做示例子
#选取的特征值为前一日的股票开盘价(open)、最高价(high)、最低价(low)、成交量(vol) 收盘价(close)，归类预测第二日的股票涨跌，涨为1、跌为0。
#可以看出来结果很一般了，这么难的分析肯定不能这么简单预测出来

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import tushare
import os
os.environ['TZ'] = 'Asia/Shanghai'


def initData():
	pro = tushare.pro_api("1bbf2b41a2180bc45022199c6ffdab5b85b07729562d189b22724b93")
	df = pro.daily(ts_code='000001.SZ', start_date='20190101', end_date='20200401', fields='trade_date,open,high,low,close,vol')
	df.to_csv('./stock_train.csv')

#获取数据
def getData():
	df = pd.read_csv('./stock_train.csv')
	data = df['close'] - df['open']
	data.index = np.arange(-1, len(data)-1)
	ret = data.map(lambda x: 1 if x>0 else 0)
	#归一化
	df = df.apply(lambda x : (x-x.mean())/x.std())
	#删除缺失值
	df['y'] = ret
	df = df.dropna()

	return df

#做散点图分析（基本上从这个图看不出有啥特征）
def imgAnalyze():
	df = getData()
	plt.subplot(1,1,1)
	plt.scatter(df[df['y']==1].loc[:,'high'], df[df['y']==1].loc[:,'vol'], c='b', label='zhang')
	plt.scatter(df[df['y']==0].loc[:,'high'], df[df['y']==0].loc[:,'vol'], c='c', label='die')
	plt.legend()
	plt.show()

#调用sklearn 逻辑回归预测
def reg():
	#以25%
	df = getData()
	x_train,x_test,y_train,y_test = train_test_split(df[['open','high', 'low', 'vol', 'close']], df['y'], test_size=0.25)
	lr = LogisticRegression()
	lr.fit(x_train, y_train)
	print(lr.coef_)	#θ1...
	print(lr.intercept_)	#θ0
	data = pd.DataFrame(columns=['real', 'predict'], index=np.arange(0,len(y_test)))
	y_hat= lr.predict(x_test)
	data['real'] = y_test.values
	data['predict'] = y_hat
	print(data.head())

#梯度下降发
def sigmoid(z):
	return 1/(1+np.exp(-z))

def predict(theta, X):
	prod = sigmoid(X*theta.T)      #把线性回归模型带入到sigmoid，获取回归概率
	return [1 if a>= 0.5 else 0 for a in prod] #设定阈值为0.5

def gradientDescent(X, y, theta, alpha, m, numIter):
	XTrans = X.T     #获取转置矩阵
	for i in range(0, numIter):    #梯度下降步数
		theta = np.matrix(theta)   #矩阵转换
		loss = np.array(predict(theta, X)) - y  #hθ(x) - y 的矩阵
		gradient = np.dot(XTrans,loss)    #因为XTrans是转置矩阵，所以结果相当求和了
		theta = theta - (alpha/m)*gradient    #求解θ
	return theta

#循环梯度下降
def selfGradient():
	df = getData()
	df.insert(0, 'x0', np.ones(len(df)))
	x_train,x_test,y_train,y_test = train_test_split(df[['x0','open','high', 'low', 'vol', 'close']], df['y'], test_size=0.25)
	theta = np.ones(6)		#初始化theta
	alpha = 0.0001
	m = len(x_train)
	numIter = 1000
	theta = gradientDescent(x_train.values, y_train.values, theta, alpha, m, numIter)
	print(theta)
	y_hat = predict(theta, x_test.values)
	data = pd.DataFrame(columns=['real', 'predict'], index=np.arange(0,len(y_test)))
	data['real'] = y_test.values
	data['predict'] = y_hat
	print(data.head())

if __name__ == '__main__':
	reg()



