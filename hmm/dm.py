#动态规划，大规模的问题，能由小问题推出;
#建立动态方程 f(x) = g(f(x-1)), 存储中间解, 按顺序从小往大算

#动态规划：求最长子序列
def lis(d):
	b = []
	for i,v in enumerate(d):
		if i==0:
			b.append(v)
		if d[i] < b[-1]:
			for k,bv in enumerate(b):
				if d[i] > v:
					b = b[0:k]
		if d[i] > b[-1]:
			b.append(d[i])
	return b

#动态规划：斐波那契数列
def fib(n):
	#缓存结果
	results = list(range(n+1))

	for i in range(n+1):
		if i<2 :
			results[i] = i
		else:
			results[i] = results[i-1] + results[i-2]

	return results[-1]

if __name__ == '__main__':
	print(lis([1,3,5,6,2,4,7,9]))
	print(fib(100))