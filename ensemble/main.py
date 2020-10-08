import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier

pd.set_option('display.max_columns', None)

def getTrainData():
	df = pd.read_csv('./data/happiness_train_abbr.csv')
	df.drop(['id', 'survey_time'], axis=1, inplace=True)
	return df.dropna()


if __name__ == '__main__':
	df = getTrainData()
	x = df.drop(['happiness'], axis=1)
	x = preprocessing.scale(x)
	y = df['happiness']
	x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25,random_state=0)

	lr = LogisticRegression(random_state=0, max_iter=1500, warm_start=True)
	lr.fit(x_train, y_train)

	y_hat = lr.predict(x_test)
	print(np.mean(y_hat==y_test))

	ab = AdaBoostClassifier()
	ab.fit(x_train, y_train)
	y_ab = ab.predict(x_test)
	print(np.mean(y_ab==y_test))