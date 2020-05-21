import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection #模型比较和选择包
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def lineSvm():
	#设置相同的seed，则每次生成的随机数都相同
	np.random.seed(0)

	#np.r_按列连接两个矩阵、要求列数相同
	#np.c_按行连接两个矩阵、要求行数相同
	#np.random.randn(a,b)生成0，1之间，包含0，a行b列的数组
	#生成随机数
	X = np.r_[np.random.randn(38, 2)-[2,2], np.random.randn(42,2)+[2,2]]
	y = [0]*38 + [1]*42

	#散点图示
	# fig,ax = plt.subplots(figsize=(8,6))
	# ax.scatter(X[0:40, 1], X[0:40, 0], s=30, c='b', marker='o', label='y=0')
	# ax.scatter(X[40:80, 1], X[40:80, 0], s=30, c='b', marker='x', label='y=1')
	# ax.legend()
	# plt.show()

	clf = SVC(kernel='linear')
	clf.fit(X, y)		#模型训练
	y_hat = clf.predict(X)
	print("评分：", clf.score(X, y_hat))

	w = clf.coef_[0]		#模型的w
	a = -w[0]/w[1]		#获取斜率
	xx = np.linspace(-5, 5)
	yy = a*xx - (clf.intercept_[0])/w[1]	#超平面yy

	b = clf.support_vectors_[0]		#获取支持向量的第一列
	yy_down = a*xx+(b[1] - a*b[0])	#生成下方的支撑向量
	b = clf.support_vectors_[-1]
	yy_up = a*xx + (b[1] - a*b[0])	#生成上方yy

	plt.plot(xx, yy, 'k-')
	plt.plot(xx, yy_down, 'k--')
	plt.plot(xx, yy_up, 'k--')

	#绘制支撑向量的散点
	plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='black', s=30, facecolors='none', marker='*')	

	plt.scatter(X[0:40, 1], X[0:40, 0], s=30, c='b', marker='o', label='y=0')
	plt.scatter(X[40:80, 1], X[40:80, 0], s=30, c='b', marker='x', label='y=1')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	lineSvm()

