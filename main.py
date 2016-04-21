import operator

from deap import creator
from deap import base
from deap import gp
from deap import tools
from deap import algorithms

import numpy as np
from numpy import linalg as la

from sklearn.metrics import adjusted_rand_score

import utils
from ART import OnlineFuzzyART
from data import XCSVFileReader

__author__ = 'Islam Elnabarawy'

# FuzzyART algorithm parameters
rho, alpha, beta = 0.555584128569494, 0.7568376651388451, 0.5837066700986062


def max_norm(x):
    # noinspection PyTypeChecker
    return la.norm(x, ord=1)


def fuzzy_and(x, y):
    return np.min(np.array([x, y]), 0)


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSetTyped("main", [np.ndarray, np.ndarray, float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(max_norm, [np.ndarray], float)
pset.addPrimitive(fuzzy_and, [np.ndarray, np.ndarray], np.ndarray)

pset.renameArguments(ARG0="A")
pset.renameArguments(ARG1="w_j")
pset.renameArguments(ARG2="alpha")

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

with XCSVFileReader('data/iris.data') as reader:
    dataset = np.zeros((len(reader), reader.num_fields-1))
    labels = np.zeros((len(reader)))
    data_ranges = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
    for ix, row in enumerate(reader):
        pattern = np.array(row[:4], dtype=float)
        for i in range(len(pattern)):
            pattern[i] = utils.scale_range(pattern[i], data_ranges[i])
        dataset[ix, :] = pattern
        labels[ix] = row[4]


def eval_individual(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Create a FuzzyART network with the individual's category choice function
    fa = OnlineFuzzyART(rho, alpha, beta, len(data_ranges), choice_fn=func)
    # Run the clustering on the dataset and find the clusters
    iterations, clusters = fa.run_batch(dataset, max_epochs=100, seed=100)
    # return the adjusted rand index for the result
    return adjusted_rand_score(labels, clusters),

toolbox.register("evaluate", eval_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
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

pop = toolbox.population(n=30)
hof = tools.HallOfFame(5)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)
