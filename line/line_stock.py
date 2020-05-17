import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import random

#使用linearReg函数预测
def linearReg():
	df = pd.read_csv('./stock_train.csv')
	#定义预测目标值
	target = df['close']
	#定义预测特征值
	feature = df[['open', 'high', 'low', 'volume']]
	#划分训练集与测试集
	x_train,x_test,y_train,y_test = train_test_split(feature, target, test_size=0.25)

	lr = LinearRegression()
	lr.fit(x_train, y_train)

	y_hat = lr.predict(x_test)

	print("r2_score:",r2_score(y_test, y_hat))
	print("MAE:", metrics.mean_absolute_error(y_test, y_hat))
	print("MSE:", metrics.mean_squared_error(y_test, y_hat))

	plt.scatter(x_test['open'], y_test, c='red',alpha = 0.8)
	plt.scatter(x_test['open'], y_hat, c='#0909ef', alpha = 0.4)
	plt.show()

#手动推导梯度下降法
def gradientDescent():
	rate = 0.0001	#定义学习率
	nums = 1000		#梯度下降步数
	r = 0.000001		#正则化系数

	df = pd.read_csv('./stock_train.csv')
	#定义预测目标值
	target = df['close']
	#数据归一化
	target = (target-target.min())/(target.max()-target.min())
	#定义预测特征值
	feature = df[['open', 'high', 'low', 'volume']]
	#数据归一化
	feature = (feature-feature.min())/(feature.max()-feature.min())
	feature.insert(0, 'num', np.ones(len(feature)))

		#划分训练集与测试集
	x_train,x_test,y_train,y_test = train_test_split(feature, target, test_size=0.25)
	theta = np.zeros((1, 5))
	temp = np.zeros(theta.shape)		#初始化矩阵
	parameters = 5    #计算需要求解的thata参数个数
	m = len(x_train)
	for i in range(nums):
		#随机梯度下降法
		index = np.random.randint(0, len(x_train))
		X = x_train.sample(n=index).values
		Y = y_train.sample(n=index).values

		error = np.dot(X,theta.T) - Y 		#对应函数hθ(x) - y
		for j in range(parameters):
			#对应公式 θ(j) = θ(j)*(1-λ*α/m) - α/m*∑(hθ(x) - y)*x(j), 这里引入了正则化系数
			t = np.multiply(error, X[:, j]) #数惩
			temp[0, j] = theta[0, j]*(1-r*(rate/m)) - (rate/m)*np.sum(t)
		theta = temp

	print(theta)

	y_hat = np.dot(x_test.values, theta.T)

	print("r2_score:",r2_score(y_test, y_hat))
	print("MAE:", metrics.mean_absolute_error(y_test, y_hat))
	print("MSE:", metrics.mean_squared_error(y_test, y_hat))
	# plt.scatter(x_test['open'], y_test, c='red',alpha = 0.8)
	# plt.scatter(x_test['open'], y_hat, c='#0909ef', alpha = 0.4)
	# plt.show()

#最小二乘法
def LeastSquares():
	df = pd.read_csv('./stock_train.csv')
	#定义预测目标值
	target = df['close']
	target = (target-target.min())/(target.max()-target.min())
	#定义预测特征值
	feature = df[['open', 'high', 'low', 'volume']]
	feature = (feature-feature.min())/(feature.max()-feature.min())
	#补齐 X参数theta同纬度
	feature.insert(0, 'num', np.ones(len(feature)))
	#划分训练集与测试集
	x_train,x_test,y_train,y_test = train_test_split(feature, target, test_size=0.25)

	X = x_train.values
	Y = y_train.values
	#公式2.6
	theta = np.dot(np.dot(np.linalg.inv((np.dot(X.T,X))), X.T), Y)
	print(theta)

	y_hat = np.dot(x_test.values, theta.T)

	print("r2_score:",r2_score(y_test, y_hat))
	print("MAE:", metrics.mean_absolute_error(y_test, y_hat))
	print("MSE:", metrics.mean_squared_error(y_test, y_hat))
	plt.scatter(x_test['open'], y_test, c='red',alpha = 0.8)
	plt.scatter(x_test['open'], y_hat, c='#0909ef', alpha = 0.4)
	plt.show()

if __name__ == '__main__':
	#linearReg()
	# LeastSquares()
	gradientDescent()
