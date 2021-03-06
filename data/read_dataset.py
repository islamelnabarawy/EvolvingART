import csv

import numpy as np
import arff

import utils

__author__ = 'Islam Elnabarawy'


def read_dataset(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        dataset = np.array([row for row in reader], dtype=float)
        labels = dataset[:, -1]
        dataset = dataset[:, :-1]
        data_ranges = list(zip(dataset.min(axis=0), dataset.max(axis=0)))
        for ix, pattern in enumerate(dataset):
            for i in range(len(pattern)):
                pattern[i] = utils.scale_range(pattern[i], data_ranges[i])
            dataset[ix, :] = pattern
        return dataset, labels


def read_arff_dataset(filename):
    arff_data = arff.load(filename)
    dataset = np.array([[float(val) for val in row] for row in arff_data], dtype=float)
    labels = dataset[:, -1]
    dataset = dataset[:, :-1]
    return dataset, labels
