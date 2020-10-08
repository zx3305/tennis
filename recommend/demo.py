from sklearn.feature_extraction.text import TfidfVectorizer

#TF-IDF(Term Frequency-inverse Document Frequency) 一种针对关键字的统计方法，用于评估一个词对一个文件集或者一个语料库的重要程度。
#TF = （某词在文档中出现的次数/文档的总词量）
#IDF = log(语料库中文档总数/(包含该词的文档数+1)),IDF越大说明该词有很强的区分能力
#TF-IDF = TF * IDF 值越大表明该特征词对文本的重要性越大

#https://blog.csdn.net/blmoistawinde/article/details/80816179
def tf_idf_test():
	corpus = [
		'This is the first document.',
		'This document is the second document.',
		'And this is the third one.',
		'Is this the first document?',
	]
	vectorizer = TfidfVectorizer()
	tdm = vectorizer.fit_transform(corpus)
	print(tdm)
	print(tdm.todense())
	space = vectorizer.vocabulary_
	print(space)

if __name__ == '__main__':
	tf_idf_test()