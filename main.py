import argparse
import operator
import multiprocessing
import logging

from deap import creator
from deap import base
from deap import gp
from deap import tools
from deap import algorithms

import numpy as np
from numpy import linalg as la

from sklearn.metrics import adjusted_rand_score

from data.read_dataset import read_arff_dataset
from ART import OnlineFuzzyART

__author__ = 'Islam Elnabarawy'

# FuzzyART algorithm parameters
RHO, ALPHA, BETA = 0.6, 0.05, 0.95
# rho, alpha, beta = 0.5173929115731474, 0.47460905154087896, 0.6250151337909732   # iris.data
# rho, alpha, beta = 0.4249555132101839, 0.0011891228422072908, 0.5315274236032594   # glass.data

NUM_FOLDS = 10


def evaluate(individual, compile, dataset, labels, rho, alpha, beta):
    # Transform the tree expression in a callable function
    func = compile(expr=individual)
    # print("Evaluating individual: %s" % individual)
    # Create a FuzzyART network with the individual's category choice function
    fa = OnlineFuzzyART(rho, alpha, beta, dataset.shape[1], choice_fn=func)
    # Run the clustering on the dataset and find the clusters
    iterations, clusters = fa.run_batch(dataset, max_epochs=10)
    # return the adjusted rand index for the result
    performance = adjusted_rand_score(labels, clusters)
    # log the result
    # print("\tIterations: %02d\tPerformance: %s" % (iterations, performance))
    return performance,


def max_norm(x):
    # noinspection PyTypeChecker
    return la.norm(x, ord=1)


def fuzzy_and(x, y):
    return np.min(np.array([x, y]), 0)


def fuzzy_or(x, y):
    return np.max(np.array([x, y]), 0)


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def get_primitive_set():
    pset = gp.PrimitiveSetTyped("main", [np.ndarray, np.ndarray, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protected_div, [float, float], float)
    # pset.addPrimitive(operator.neg, [float], float)
    pset.addPrimitive(max_norm, [np.ndarray], float)
    pset.addPrimitive(fuzzy_and, [np.ndarray, np.ndarray], np.ndarray)
    pset.addPrimitive(fuzzy_or, [np.ndarray, np.ndarray], np.ndarray)

    pset.renameArguments(ARG0="A")
    pset.renameArguments(ARG1="w_j")
    pset.renameArguments(ARG2="alpha")

    return pset

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def run_fold(dataset_name, fold_index, rho, alpha, beta):

    train_file = 'data/crossvalidation/{}/{}.train.arff'.format(dataset_name, fold_index)
    test_file = 'data/crossvalidation/{}/{}.test.arff'.format(dataset_name, fold_index)

    pset = get_primitive_set()

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    train_dataset, train_labels = read_arff_dataset(train_file)
    test_dataset, test_labels = read_arff_dataset(test_file)

    toolbox.register("evaluate", evaluate,
                     compile=toolbox.compile, dataset=train_dataset,
                     labels=train_labels, rho=rho, alpha=alpha, beta=beta)
    toolbox.register("select", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(3)
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 100, 20, 0.95, 0.05, 500,
                                         stats=mstats, halloffame=hof, verbose=True)

    print("Hall of fame:")
    for expr in hof:
        print('\t', expr, expr.fitness)

    print("Training set fitness:")
    for expr in hof:
        train_fitness = evaluate(expr, toolbox.compile, train_dataset, train_labels, rho, alpha, beta)
        print('\t', expr, train_fitness)

    print("Test set fitness:")
    for expr in hof:
        test_fitness = evaluate(expr, toolbox.compile, test_dataset, test_labels, rho, alpha, beta)
        print('\t', expr, test_fitness)

    return pop, hof, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=['wine', 'iris', 'glass'],
                        help="The index of the fold to evaluate")
    parser.add_argument("fold", type=int, choices=range(NUM_FOLDS),
                        help="The index of the fold to evaluate")
    parser.add_argument("--rho", type=float, required=False, default=RHO)
    parser.add_argument("--alpha", type=float, required=False, default=ALPHA)
    parser.add_argument("--beta", type=float, required=False, default=BETA)
    args = parser.parse_args()

    print("Dataset: {}".format(args.dataset))
    print("Fold: {}/{}".format(args.fold+1, NUM_FOLDS))
    print("Parameters:\n\trho = {}\n\talpha = {}\n\tbeta = {}".format(args.rho, args.alpha, args.beta))
    run_fold(args.dataset, args.fold, args.rho, args.alpha, args.beta)
    print("\nFold {}/{} done.\n".format(args.fold+1, NUM_FOLDS))


if __name__ == '__main__':
    main()
