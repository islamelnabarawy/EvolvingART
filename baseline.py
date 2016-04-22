import multiprocessing

import numpy as np
from sklearn.metrics import adjusted_rand_score

from data import read_dataset
from ART import OnlineFuzzyART

__author__ = 'Islam Elnabarawy'

rho, alpha, beta = 0.5173929115731474, 0.47460905154087896, 0.6250151337909732   # iris.data
# rho, alpha, beta = 0.4249555132101839, 0.0011891228422072908, 0.5315274236032594     # glass.data

data_file = 'data/iris.data'
# data_file = 'data/glass.data'

dataset, labels = read_dataset(data_file)


def evaluate(index):
    fa = OnlineFuzzyART(rho, alpha, beta, dataset.shape[1])
    iterations, clusters = fa.run_batch(dataset, max_epochs=100)
    performance = adjusted_rand_score(labels, clusters)
    return index, iterations, fa.num_clusters, performance


def main():
    pool = multiprocessing.Pool()
    results = pool.map(evaluate, range(100))
    print('%-4s\t%-4s\t%-8s\t%s' % ('Idx', 'Iter', 'Clusters', 'Performance'))
    print('-'*50)
    for row in results:
        print('%-4s\t%-4s\t%-8s\t%s' % row)
    print('-'*50)
    avg = np.array(results, dtype=float)[:, 1:].mean(axis=0)
    print('Avg:\t%-4s\t%-8s\t%s' % tuple(avg))

if __name__ == '__main__':
    main()
