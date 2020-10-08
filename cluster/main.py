import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

#生成聚类数据
#n_samples 生成集合长度、n_features X的维度、centers 聚类中心点、数据集的标准差
X,y = ds.make_blobs(n_samples=100,n_features=2,centers=[[1,1], [-3,3]],cluster_std=[1,2], random_state=0)

#数据压缩
X = StandardScaler().fit_transform(X)

#密度聚集算法
# eps领域大小 min_samples 核心对象需要的参数 metric 'euclidean'欧式距离 'manhattan'曼哈顿距离 
model = DBSCAN(eps=.4, min_samples=10, metric='euclidean')
model.fit(X)
culust = []
for v in model.labels_:
	culust.append(abs(v))

print("聚类成功率:", np.mean(y==culust))
