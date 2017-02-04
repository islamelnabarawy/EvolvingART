import argparse
import random
import multiprocessing
import pickle

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np

from sklearn.metrics import adjusted_rand_score

from ART import OnlineFuzzyART
from data.read_dataset import read_arff_dataset

__author__ = 'Islam Elnabarawy'

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate(individual, dataset, labels):
    fa = OnlineFuzzyART(*individual, dataset.shape[1])
    iterations, clusters = fa.run_batch(dataset, max_epochs=10)
    return adjusted_rand_score(labels, clusters),


def hypertune(dataset_name, npop, ngen, cxpb, mutpb, indpb, mu, lambda_):
    toolbox = base.Toolbox()

    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    data_file = 'data/crossvalidation/{}_norm.arff'.format(dataset_name)
    dataset, labels = read_arff_dataset(data_file)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=indpb)
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", evaluate, dataset=dataset, labels=labels)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(5)
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                                         stats=mstats, halloffame=hof, verbose=True)

    for expr in hof:
        print(expr, expr.fitness)

    return hof, pop, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=['wine', 'iris', 'glass'],
                        help="The index of the fold to tune on")
    parser.add_argument("output", help="Name of the output file to save the results to")
    args = parser.parse_args()

    print("Dataset: {}".format(args.dataset))

    params = {
        'dataset_name': args.dataset,
        'npop': 100,
        'ngen': 500,
        'cxpb': 0.95,
        'mutpb': 0.05,
        'indpb': 0.05,
        'mu': 100,
        'lambda_': 20
    }
    hof, pop, log = hypertune(**params)
    with open(args.output, "wb") as file_out:
        pickle.dump(dict(population=pop, halloffame=hof, logbook=log, params=params), file_out)


if __name__ == '__main__':
    main()
