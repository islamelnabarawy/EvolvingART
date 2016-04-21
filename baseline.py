import numpy as np

from sklearn.metrics import adjusted_rand_score

import utils
from ART import OnlineFuzzyART
from data import XCSVFileReader

__author__ = 'Islam Elnabarawy'


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

fa = OnlineFuzzyART(0.4, 0.001, 0.9, len(data_ranges))
iterations, clusters = fa.run_batch(dataset, max_epochs=100, seed=100)

performance = adjusted_rand_score(labels, clusters)

print('FuzzyART found %s clusters.' % fa.num_clusters)
print('Adjusted Rand Index: %s' % performance)
