import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sympy as sp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
线性回归，单变量实例和最小二乘法解法。
'''

#构造模拟数据
def initData():
	x = sp.symbols('x')
	a = sp.symbols('a')
	b = sp.symbols('b')

	fx = a*x + b
	xArr = []
	yArr = []
	for i in np.arange(1, 90, 0.3):
		subs = {x:i+random.uniform(-0.3,0.5), a:4+random.uniform(-0.3,0.3), b:5+random.uniform(-0.2, 0.2)}
		xArr.append(subs[x])
		yArr.append(fx.evalf(subs=subs))
	return xArr,yArr

#图片分析观察数据
def imgShow():
	xArr, yArr = initData()
	plt.scatter(xArr, yArr,  s=20, c="#ff1212", marker='.')
	plt.show()

#采用LinearRegression求解
def linearReg(x_train,x_test,y_train,y_test):
	lr = LinearRegression()			#选定回归算法
	lr.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))		#使用回归算法进行参数求解，求解y=ax+b 的a与b

	print("系数a=", lr.coef_)
	print("系数b=", lr.intercept_)

	y_hat = lr.predict(x_test.reshape(-1, 1))		#预测值

	t = np.arange(len(x_test))
	plt.plot(t, y_hat, linewidth=1, label='y_train')
	plt.plot(t, y_test, linewidth=1, label='y_test')
	plt.legend()
	plt.show()

#最小二乘法推导
def linefit(x_train, x_test, y_train, y_test):
	x,y = x_train, y_train
	N = len(x)

	sx,sy,sxx,sxy = 0,0,0,0 #设置初始值
	for i in range(0, N):
		sx += x[i]
		sy += y[i]
		sxx += x[i] * x[i]
		sxy += x[i] * y[i]
	a = (sy*sx/N - sxy)/(sx*sx/N - sxx)    #求解系数a
	b = (sy - a*sx) / N 		#求解系数b,公式1.5
	y_hat = a*x_test + b

	t = np.arange(len(x_test))

	plt.plot(t, y_hat, linewidth=1, label='y_train')
	plt.plot(t, y_test, linewidth=1, label='y_test')
	plt.legend()
	plt.show()



if __name__=='__main__':
	xArr, yArr = initData()
	xArr = np.array(xArr)
	yArr = np.array(yArr)

	#以25%的数据构造测试样本,其余构造训练样本
	x_train,x_test,y_train,y_test = train_test_split(xArr, yArr, test_size=0.25)

	# imgShow()
	# linearReg(x_train,x_test,y_train,y_test)
	linefit(x_train,x_test,y_train,y_test)


