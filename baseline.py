from sklearn.metrics import adjusted_rand_score

from data import read_dataset
from ART import OnlineFuzzyART

__author__ = 'Islam Elnabarawy'

rho, alpha, beta = 0.5342327836238868, 0.3119799068437582, 0.5472496744945247

dataset, labels = read_dataset('data/iris.data')
fa = OnlineFuzzyART(rho, alpha, beta, dataset.shape[1])
iterations, clusters = fa.run_batch(dataset, max_epochs=100, seed=100)
performance = adjusted_rand_score(labels, clusters)

print('FuzzyART found %s clusters.' % fa.num_clusters)
print('Adjusted Rand Index: %s' % performance)
