import numpy as np
import sklearn.datasets as ds

#生成聚类数据
#n_samples 生成集合长度、n_features X的维度、centers 聚类中心点、数据集的标准差
X,y = ds.make_blobs(n_samples=2,n_features=10,centers=[[1,1], [-3,3]],cluster_std=[1,2], random_state=0)

print("欧式距离：",np.sqrt(np.sum(np.square(X[0]-X[1]))))
print("曼哈顿距离：", np.sum(np.abs(X[0]-X[1])))
print("切比雪夫距离：", np.max(np.abs(X[0]-X[1])))
print("马氏距离：", np.sum(np.sqrt((X[0]-X[1])*np.abs(np.linalg.inv(np.cov(X[0],X[1])))*(X[0]-X[1]))))
print("夹角余弦：", (np.sum(X[0].T*X[1]))/(np.sqrt(np.sum(X[0].T*X[0]))*np.sqrt(np.sum(X[1].T*X[1]))))
print("person相关系数:", np.corrcoef(X))