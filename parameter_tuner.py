import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np

from sklearn.metrics import adjusted_rand_score

import utils
from ART import OnlineFuzzyART
from data import XCSVFileReader

__author__ = 'Islam Elnabarawy'

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

IND_SIZE = 3

toolbox = base.Toolbox()

toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

with XCSVFileReader('data/iris.data') as reader:
    dataset = np.zeros((len(reader), reader.num_fields-1))
    labels = np.zeros((len(reader), ))
    data_ranges = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
    for ix, row in enumerate(reader):
        pattern = np.array(row[:4], dtype=float)
        for i in range(len(pattern)):
            pattern[i] = utils.scale_range(pattern[i], data_ranges[i])
        dataset[ix, :] = pattern
        labels[ix] = row[4]


def evaluate(individual):
    fa = OnlineFuzzyART(*individual, len(data_ranges))
    iterations, clusters = fa.run_batch(dataset, max_epochs=100, seed=100)
    return adjusted_rand_score(labels, clusters),

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
mstats = tools.MultiStatistics(fitness=stats_fit)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)


def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(5)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)

    print(hof[0])

    return hof, pop, log


if __name__ == '__main__':
    main()
