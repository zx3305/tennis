import pandas as pd
import numpy as np


#把数据转化为csv格式数据
def dataToCsv():
	moviesStr = "MovieID,Title,Genres\n"
	with open('./data/movies.dat',  encoding="latin-1") as file:
		lines = file.readlines()
		for line in lines:
			moviesStr = moviesStr + line.replace(',', '').replace('::',',')
	with open('./data/movies.csv', 'w+', encoding = 'utf-8') as f:
		f.write(moviesStr)

	ratingStr = "UserID,MovieID,Rating,Timestamp\n"
	with open('./data/ratings.dat',  encoding="latin-1") as file:
		lines = file.readlines()
		for line in lines:
			ratingStr = ratingStr + line.replace(',', '').replace('::',',')
	with open('./data/ratings.csv', 'w+', encoding = 'utf-8') as f:
		f.write(ratingStr)

	usersStr = "UserID,Gender,Age,Occupation,Zip-code\n"
	with open('./data/users.dat',  encoding="latin-1") as file:
		lines = file.readlines()
		for line in lines:
			usersStr = usersStr + line.replace(',', '').replace('::',',')
	with open('./data/users.csv', 'w+', encoding = 'utf-8') as f:
		f.write(usersStr)

#构建(电影,用户)矩阵数据
def moveisUserMatrix():
	#取10万数据进行训练, 并过滤用户记录小于600的数据
	df = pd.read_csv('./data/ratings.csv')
	df.drop(['Timestamp'], axis=1, inplace=True)
	df[['UserID','MovieID','Rating']] = df[['UserID','MovieID','Rating']].astype("int32")
	userDf = df.groupby('UserID',axis = 0)['UserID'].count().sort_values(ascending=False)
	userDf = userDf[userDf>600]
	userList = userDf.index
	df = df[df['UserID'].isin(userList)]
	df[['UserID','MovieID','Rating']] = df[['UserID','MovieID','Rating']].astype("int32")
	print(df.info())

	#抽取50%的数据训练
	df = df.sample(frac=.5, replace=False, random_state=0)

	#找到所有用户
	userList = np.unique(df['UserID'].values)
	#找到所有的电影
	movesList = np.unique(df['MovieID'].values)

	result = []
	i = 0
	print("共:"+str(len(movesList)))
	f = open("./data/tmp.log", "a+")
	f2 = open("./data/tmp.log", "r")
	lines = f2.readlines()

	print(len(lines))

	fileNum = len(lines)

	#构建矩阵
	for movesId in movesList:
		movesRow = []
		if i < fileNum:
			movesRow = lines[i].replace('\n', '').split(',')
		else:
			for userId in userList:
				qs = 'UserID=='+str(userId)+' & MovieID=='+str(movesId)
				tmpdf = df.query(qs)
				if tmpdf.empty:
					movesRow.append(str(0))
				else:
					movesRow.append(str(tmpdf["Rating"].mean()))
			f.write(','.join(movesRow)+"\n")
		i = i + 1
		result.append(movesRow)
		print(str(i)+":处理完成")

	ret = pd.DataFrame(result, columns=userList, index=movesList)
	ret.to_csv("./data/baseItemMatrix.csv")
	f.close()

#缺失值处理
#计算有值个数
def dataClean():
	df = pd.read_csv('./data/baseItemMatrix.csv')
	n = len(df.index)
	for i, s in df.iterrows():	
		print(i)
		s = s[1:]
		n = s[s.values >0].count()
		mean = s.sum()/n
		df.iloc[i,1:] = s.apply(lambda x : round(x-mean,2) if x>0 else 0)
	df.to_csv("./data/baseItemMatrix_2.csv")


#计算物品间的相似度，这里用pearson相似度
def itemSimilar():
	df = pd.read_csv('./data/baseItemMatrix_2.csv')
	df.drop(columns=['Unnamed: 0'],inplace=True)
	moveIds = df["moveId"].values
	n = len(moveIds)
	np.seterr(divide='ignore',invalid='ignore')
	f = open("./data/tmp2.log", "a+")
	f2 = open("./data/tmp2.log", "r")
	lines = f2.readlines()
	fileNum = len(lines)
	#初始化矩阵 n*n
	initM = np.zeros((n,n))
	for k,v in enumerate(moveIds):
		print(k)
		if k < fileNum:
			initM[k] = lines[k].replace('\n', '').split(',')
		else:
			for k2,v2 in enumerate(moveIds):
				if v == v2:
					initM[k][k2] = str(1)
				else:
					initM[k][k2] = str(round(np.corrcoef(df[df['moveId']==v].values[0][1:], df[df['moveId']==v2].values[0][1:])[0][1],4))

			f.write(','.join('%s' %id for id in initM[k])+"\n")
	f.close()
	ret = pd.DataFrame(initM, columns=moveIds, index=moveIds)
	df.to_csv("./data/baseItemMatrix_3.csv")


#推荐方法
#1. 找到用户看过的电影
#2. 找到这些电影用户未看过的最相似的电影
#3. 估算用户可能的评分，大于4分的推荐 
def recommend(userId):
	pass

if __name__ == '__main__':
	itemSimilar()




