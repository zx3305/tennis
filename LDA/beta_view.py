
import numpy as np
from scipy.stats import gamma
from matplotlib import pyplot as plt
from scipy.stats import beta

# Gamma分布（Gamma Distribution）
def gama_distribution():
	alpha_values = [1, 2, 3, 3, 3]
	beta_values = [0.5, 0.5, 0.5, 1, 2]
	color = ['b','r','g','y','m']
	x = np.linspace(1E-6, 10, 1000)

	fig, ax = plt.subplots(figsize=(12, 8))

	for k, t, c in zip(alpha_values, beta_values, color):
	    dist = gamma(k, 0, t)
	    plt.plot(x, dist.pdf(x), c=c, label=r'$\alpha=%.1f,\beta=%.1f$' % (k, t))

	plt.xlim(0, 10)
	plt.ylim(0, 2)

	plt.xlabel('$x$')
	plt.ylabel(r'$p(x|\alpha,\beta)$')
	plt.title('Gamma Distribution')

	plt.legend(loc=0)
	plt.show()

# Beta 分布（Beta distribution）
def beta_distribution():
	alpha_values = [1/3,2/3,1,1,2,2,4,10,20]
	beta_values = [1,2/3,3,1,1,6,4,30,20]
	colors =  ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
	           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
	x = np.linspace(0, 1, 1002)[1:-1]

	fig, ax = plt.subplots(figsize=(14,9))

	for a, b, c in zip(alpha_values, beta_values, colors):
	    dist = beta(a, b)
	    plt.plot(x, dist.pdf(x), c=c,label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))

	plt.xlim(0, 1)
	plt.ylim(0, 6)

	plt.xlabel('$x$')
	plt.ylabel(r'$p(x|\alpha,\beta)$')
	plt.title('Beta Distribution')

	ax.annotate('Beta(1/3,1)', xy=(0.014, 5), xytext=(0.04, 5.2),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	ax.annotate('Beta(10,30)', xy=(0.276, 5), xytext=(0.3, 5.4),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	ax.annotate('Beta(20,20)', xy=(0.5, 5), xytext=(0.52, 5.4),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	ax.annotate('Beta(1,3)', xy=(0.06, 2.6), xytext=(0.07, 3.1),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	ax.annotate('Beta(2,6)', xy=(0.256, 2.41), xytext=(0.2, 3.1),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	ax.annotate('Beta(4,4)', xy=(0.53, 2.15), xytext=(0.45, 2.6),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	ax.annotate('Beta(1,1)', xy=(0.8, 1), xytext=(0.7, 2),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	ax.annotate('Beta(2,1)', xy=(0.9, 1.8), xytext=(0.75, 2.6),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	ax.annotate('Beta(2/3,2/3)', xy=(0.99, 2.4), xytext=(0.86, 2.8),
	            arrowprops=dict(facecolor='black', arrowstyle='-'))
	#plt.legend(loc=0)
	plt.show()

if __name__ == '__main__':
	beta_distribution()
