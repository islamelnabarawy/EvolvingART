import operator

from deap import creator
from deap import base
from deap import gp
from deap import tools
from deap import algorithms

import numpy as np
from numpy import linalg as la

from sklearn.metrics import adjusted_rand_score

from data import read_dataset
from ART import OnlineFuzzyART

__author__ = 'Islam Elnabarawy'

# FuzzyART algorithm parameters
rho, alpha, beta = 0.5342327836238868, 0.3119799068437582, 0.5472496744945247   # iris.data
# rho, alpha, beta = 0.4249555132101839, 0.0011891228422072908, 0.5315274236032594   # glass.data

data_file = 'data/iris.data'
# data_file = 'data/glass.data'


def max_norm(x):
    # noinspection PyTypeChecker
    return la.norm(x, ord=1)


def fuzzy_and(x, y):
    return np.min(np.array([x, y]), 0)


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSetTyped("main", [np.ndarray, np.ndarray, float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protected_div, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(max_norm, [np.ndarray], float)
pset.addPrimitive(fuzzy_and, [np.ndarray, np.ndarray], np.ndarray)

pset.renameArguments(ARG0="A")
pset.renameArguments(ARG1="w_j")
pset.renameArguments(ARG2="alpha")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

dataset, labels = read_dataset(data_file)


def eval_individual(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # print("Evaluating individual: %s" % individual)
    # Create a FuzzyART network with the individual's category choice function
    fa = OnlineFuzzyART(rho, alpha, beta, dataset.shape[1], choice_fn=func)
    # Run the clustering on the dataset and find the clusters
    iterations, clusters = fa.run_batch(dataset, max_epochs=10, seed=100)
    # return the adjusted rand index for the result
    performance = adjusted_rand_score(labels, clusters)
    # log the result
    # print("\tIterations: %02d\tPerformance: %s" % (iterations, performance))
    return performance,

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

pop = toolbox.population(n=50)
hof = tools.HallOfFame(3)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 40, stats=mstats, halloffame=hof, verbose=True)

print("Hall of fame:")
for expr in hof:
    print('\t', expr, expr.fitness)
