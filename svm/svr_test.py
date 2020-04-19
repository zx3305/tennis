import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from sklearn.metrics import r2_score

np.random.seed(0)

x = np.random.randn(80, 2)
y = x[:, 0] + 2*x[:, 1] + np.random.randn(80)

# fig,ax = plt.subplots(figsize=(8,6))
# ax.scatter(y, x[:, 0], s=30, c='b', marker='o')
# ax.scatter(y, x[:, 1], s=30, c='c', marker='^')
# plt.show()
# exit(0)

clf = SVR(kernel='linear', C=1.25)
x_tran,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25)
clf.fit(x_tran, y_train)
y_hat = clf.predict(x_test)

print("得分:", r2_score(y_test, y_hat))

r = len(x_test) + 1
print(y_test)
plt.plot(np.arange(1,r), y_hat, 'go-', label="predict")
plt.plot(np.arange(1,r), y_test, 'co-', label="real")
plt.legend()
plt.show()

'''
scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, 
	vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, 
	edgecolors=None, hold=None, data=None, **kwargs)
x，y：输入数据，array_like，shape（n，）
s: 点的大小
c: 点的颜色 b-bule、c-cyan、g-green、k-black、m-magenta、r-red、w-white、y-yellow
marker：点的形状
alpha：透明度
label: 点标记

plt.plot(x, y, format_string, **kwargs)
x,y: x、y轴数据，列表或数组
format_string: 'go-'; g颜色和scatter c参数相同之处rgb颜色、o标记类型、-线的类型
**kwargs:
	color: 控制颜色, color='green'
	linestyle : 线条风格, linestyle='dashed'

'''