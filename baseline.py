import multiprocessing

import numpy as np
from sklearn.metrics import adjusted_rand_score

from data.read_dataset import read_arff_dataset
from ART import OnlineFuzzyART

__author__ = 'Islam Elnabarawy'

rho, alpha, beta = 0.5173929115731474, 0.47460905154087896, 0.6250151337909732   # iris.data
# rho, alpha, beta = 0.4249555132101839, 0.0011891228422072908, 0.5315274236032594     # glass.data

NUM_FOLDS = 10
dataset_name = 'iris'
test_file_format = 'data/crossvalidation/' + dataset_name + '/{0}.test.arff'
train_file_format = 'data/crossvalidation/' + dataset_name + '/{0}.train.arff'


def evaluate_train(index):
    dataset, labels = read_arff_dataset(train_file_format.format(index))
    fa = OnlineFuzzyART(rho, alpha, beta, dataset.shape[1])
    iterations, clusters = fa.run_batch(dataset, max_epochs=100)
    performance = adjusted_rand_score(labels, clusters)
    return index, iterations, fa.num_clusters, performance


def evaluate_test(index):
    dataset, labels = read_arff_dataset(test_file_format.format(index))
    fa = OnlineFuzzyART(rho, alpha, beta, dataset.shape[1])
    iterations, clusters = fa.run_batch(dataset, max_epochs=100)
    performance = adjusted_rand_score(labels, clusters)
    return index, iterations, fa.num_clusters, performance


def main():
    pool = multiprocessing.Pool()
    train_results = pool.map(evaluate_train, range(NUM_FOLDS))
    test_results = pool.map(evaluate_test, range(NUM_FOLDS))
    print("Training set results:")
    print_results(train_results)
    print("Testing set results:")
    print_results(test_results)


def print_results(results):
    print('%-4s\t%-4s\t%-8s\t%s' % ('Idx', 'Iter', 'Clusters', 'Performance'))
    print('-' * 50)
    for row in results:
        print('%-4s\t%-4s\t%-8s\t%s' % row)
    print('-' * 50)
    avg = np.array(results, dtype=float)[:, 1:].mean(axis=0)
    print('Avg:\t%-4s\t%-8s\t%s' % tuple(avg))


if __name__ == '__main__':
    main()
