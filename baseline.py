import argparse
import multiprocessing

import numpy as np
from sklearn.metrics import adjusted_rand_score
from prettytable import PrettyTable

from deap import gp

from data.read_dataset import read_arff_dataset
from ART import OnlineFuzzyART
from main import get_primitive_set

__author__ = 'Islam Elnabarawy'

RHO, ALPHA, BETA = 0.6, 0.05, 0.95
DEFAULT_CHOICE_FN = 'protected_div(max_norm(fuzzy_and(A, w_j)), add(alpha, max_norm(w_j)))'
# rho, alpha, beta = 0.5173929115731474, 0.47460905154087896, 0.6250151337909732       # iris.data
# rho, alpha, beta = 0.4249555132101839, 0.0011891228422072908, 0.5315274236032594     # glass.data

NUM_FOLDS = 10
TEST_FILE_FORMAT = 'data/crossvalidation/{}/{}.test.arff'
TRAIN_FILE_FORMAT = 'data/crossvalidation/{}/{}.train.arff'


def evaluate_train(args):
    ix, filename, rho, alpha, beta, ccf = args
    dataset, labels = read_arff_dataset(filename)
    fa = OnlineFuzzyART(rho, alpha, beta, dataset.shape[1], choice_fn=get_choice_fn(ccf))
    iterations, clusters = fa.run_batch(dataset, max_epochs=10)
    performance = adjusted_rand_score(labels, clusters)
    return ix, iterations, fa.num_clusters, performance


def evaluate_test(args):
    ix, filename, rho, alpha, beta, ccf = args
    dataset, labels = read_arff_dataset(filename)
    fa = OnlineFuzzyART(rho, alpha, beta, dataset.shape[1], choice_fn=get_choice_fn(ccf))
    iterations, clusters = fa.run_batch(dataset, max_epochs=10)
    performance = adjusted_rand_score(labels, clusters)
    return ix, iterations, fa.num_clusters, performance


def get_choice_fn(expr):
    tree = gp.PrimitiveTree.from_string(expr, get_primitive_set())
    return gp.compile(tree, get_primitive_set())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=['wine', 'iris', 'glass'],
                        help="The index of the fold to evaluate")
    parser.add_argument("--rho", type=float, required=False, default=RHO)
    parser.add_argument("--alpha", type=float, required=False, default=ALPHA)
    parser.add_argument("--beta", type=float, required=False, default=BETA)
    parser.add_argument("--ccf", required=False, default=DEFAULT_CHOICE_FN)
    args = parser.parse_args()

    print("Dataset: {}".format(args.dataset))
    print("Parameters:\n\trho = {}\n\talpha = {}\n\tbeta = {}".format(args.rho, args.alpha, args.beta))
    print("Category choice function:\n\t{}".format(args.ccf))

    test_results, train_results = run(args.dataset, args.rho, args.alpha, args.beta, args.ccf)
    print("Results:")
    print_results(train_results, test_results)


def run(dataset, rho, alpha, beta, ccf):
    train_filenames = [(ix, TRAIN_FILE_FORMAT.format(dataset, ix), rho, alpha, beta, ccf)
                       for ix in range(NUM_FOLDS)]
    test_filenames = [(ix, TEST_FILE_FORMAT.format(dataset, ix), rho, alpha, beta, ccf)
                      for ix in range(NUM_FOLDS)]
    try:
        pool = multiprocessing.Pool()
        map_fn = pool.map
    except:
        def map_fn(x, y):
            return list(map(x, y))
    train_results = map_fn(evaluate_train, train_filenames)
    test_results = map_fn(evaluate_test, test_filenames)
    return test_results, train_results


def print_results(train_results, test_results):
    x = PrettyTable()
    x.field_names = ['Idx', 'Train Iter', 'Train Clusters', 'Train Performance',
                     'Test Iter', 'Test Clusters', 'Test Performance']
    for ix in range(NUM_FOLDS):
        x.add_row(train_results[ix] + test_results[ix][1:])
    train_avg = get_average(train_results)
    test_avg = get_average(test_results)
    x.add_row(['-' * len(f) for f in x.field_names])
    x.add_row(['Avg'] + list(train_avg) + list(test_avg))
    print(x)


def get_average(results):
    avg = np.array(results, dtype=float)[:, 1:].mean(axis=0)
    return avg


if __name__ == '__main__':
    main()
