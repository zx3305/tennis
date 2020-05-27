import matplotlib.pyplot as plt
import sympy as sp
import math
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.externals.six import StringIO 
import pydotplus
# from IPython.display import Image,display

#熵函数演示，x可以看成白球数据、20-x黑球数量，y为判断摸一个球颜色事件的熵
def shan():
	x = sp.symbols('x')
	y = -(x/20)*sp.log(x/20, 2) - ((20-x)/20)*sp.log((20-x)/20, 2)

	xdata = []
	ydata = []
	for i in range(1,20):
		xdata.append(i)
		ydata.append(y.evalf(subs={x:i}))

	plt.plot(xdata, ydata)
	plt.show()

def getData():
	df = pd.read_excel("./tree.xlsx")
	df.columns = ['unkonw','y1', 'y2', 'y3', 'y4', 'x']
	df.loc[df['y1']=="好", 'y1'] = 3
	df.loc[df['y1']=="中", 'y1'] = 2
	df.loc[df['y1']=="差", 'y1'] = 1
	df.loc[df['y2']=="高", 'y2'] = 3
	df.loc[df['y2']=="中", 'y2'] = 2
	df.loc[df['y2']=="低", 'y2'] = 1	
	df.loc[df['y3']=="不必需", 'y3'] = 0
	df.loc[df['y3']=="必需", 'y3'] = 1	
	df.loc[df['y4']=="包邮", 'y4'] = 1
	df.loc[df['y4']=="不包邮", 'y4'] = 0	
	df.loc[df['x']=="不买", 'x'] = 0
	df.loc[df['x']=="买", 'x'] = 1	

	return df[['y1', 'y2', 'y3', 'y4', 'x']]

def treeClass():
	df = getData()
	x = df[['y1','y2','y3','y4']]
	y = df['x']
	x = np.array(x.values)
	y = np.array(y.values)
	clf = tree.DecisionTreeClassifier(criterion="entropy")
	clf.fit(x, y)

	dot_data = StringIO()  
	tree.export_graphviz(clf, out_file=dot_data) 
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_png("./out.png")
	

if __name__ == '__main__':
	treeClass()