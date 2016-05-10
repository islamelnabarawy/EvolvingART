import random
import multiprocessing

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np

from sklearn.metrics import adjusted_rand_score

from ART import OnlineFuzzyART
from data.read_dataset import read_arff_dataset

__author__ = 'Islam Elnabarawy'

data_file = 'data/crossvalidation/iris_norm.arff'
# data_file = 'data/glass.data'

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate(individual, dataset, labels):
    fa = OnlineFuzzyART(*individual, dataset.shape[1])
    iterations, clusters = fa.run_batch(dataset, max_epochs=10)
    return adjusted_rand_score(labels, clusters),


def main():
    toolbox = base.Toolbox()

    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    dataset, labels = read_arff_dataset(data_file)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, dataset=dataset, labels=labels)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(5)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 50, stats=mstats, halloffame=hof, verbose=True)

    for expr in hof:
        print(expr, expr.fitness)

    return hof, pop, log


if __name__ == '__main__':
    main()
