import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold


#正态分布示意图
def normalImg():
	#loc正态分布均值、scale正态分布标准差
	u = 0
	sig = 1
	u2 = 5
	sig2 = 1.2
	u3 = 10
	sig3 = 1.5
	x = np.sort(np.random.normal(loc=u,scale=sig,size=100))
	x2 = np.sort(np.random.normal(loc=u2,scale=sig2,size=100))
	x3 = np.sort(np.random.normal(loc=u3,scale=sig3,size=100))

	y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
	y2 = np.exp(-(x2 - u2) ** 2 / (2 * sig2 ** 2)) / (math.sqrt(2 * math.pi) * sig2)
	y3 = np.exp(-(x3 - u3) ** 2 / (2 * sig3 ** 2)) / (math.sqrt(2 * math.pi) * sig3)


	fig = plt.figure()
	fig = plt.figure(num=1, figsize=(15, 8),dpi=80) 


	plt.plot(x,y, label="y")
	plt.plot(x2,y2, label="y2")
	plt.plot(x3,y3, label="y3")
	#为了图示清楚增加0.02。
	plt.plot(np.sort(np.r_[x,x2,x3]),np.r_[y,y2,y3]+0.02, label="y+y2+y3")
	plt.legend()
	plt.show()

#高斯混合模型预测
def gem():
	#安德森鸢尾花卉数据集
	#样本数据为 花萼长度、花萼宽度、花瓣长度、花瓣宽度
	#target为 0: Iris setosa（山鸢尾），1: Iris virginica（北美鸢尾），2: Iris versicolor（变色鸢尾）
	iris = datasets.load_iris()
	print(len(iris.data))
	#有放回的抽样,n_splits抽出的几组数据
	skf = StratifiedKFold(n_splits=4)
	print(list(skf.split(iris.data, iris.target)))
	#取出索引
	train_index,test_index = next(iter(skf.split(iris.data, iris.target)))
	print(train_index)

	X_train = iris.data[train_index]
	y_train = iris.target[train_index]
	X_test = iris.data[test_index]
	y_test = iris.target[test_index]

	n_classes = len(np.unique(y_train))

	#高斯混合模型
	#参数解释：https://blog.csdn.net/weixin_42727069/article/details/94663195
	gm = GaussianMixture(n_components=3, covariance_type='tied', max_iter=30, random_state=0)
	#初始化均值
	# gm.means_init = np.array(X_train[y_train==i].mean(axis=0) for i in range(n_classes))

	gm.fit(X_train)
	y_train_pred  = gm.predict(X_test)
	# print(gm.means_)
	# print(gm.covariances_)
	print(gm.score(X_train))

	# colors = ['navy', 'turquoise', 'darkorange']
	# plt.figure(figsize=(6, 6))
	# plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,left=.01, right=.99)

	# for n, color in enumerate(colors):
	# 	data = iris.data[iris.target == n]
	# 	plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,label=iris.target_names[n])


	# plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
	# plt.show()

if __name__ == '__main__':
	gem()



	
