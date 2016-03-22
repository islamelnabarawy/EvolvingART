import unittest
import numpy as np

from ART import OnlineFuzzyART
from data import XCSVFileReader
import utils

__author__ = 'Islam Elnabarawy'


class TestOnlineFuzzyART(unittest.TestCase):
    def test_empty_data(self):
        fuzzyart = OnlineFuzzyART(0.15, 0.001, 1.0, 0)
        fuzzyart.run_batch(np.zeros((0, 0)), 100)
        self.assertEqual(fuzzyart.num_clusters, 0)

    def test_ml1m_user_data(self):
        with XCSVFileReader('data/users.csv') as reader:
            fa = OnlineFuzzyART(0.95, 0.001, 0.9, reader.num_fields)
            data_ranges = [(0, 1), (0, 6)] + [(0, 1)] * 21
            iterations, clusters = fa.run_online(reader, data_ranges, max_epochs=100, seed=100)
        self.assertEqual(3, iterations)
        self.assertEqual(52, fa.num_clusters)

    def test_batch_mode(self):
        data_ranges = [(0, 1), (0, 6)] + [(0, 1)] * 21

        def normalize(p):
            for i in range(len(p)):
                p[i] = utils.scale_range(p[i], data_ranges[i])
            return p

        with XCSVFileReader('data/users.csv') as reader:
            dataset = np.zeros((len(reader), reader.num_fields))
            for ix, row in enumerate(reader):
                pattern = np.array(row, dtype=float)
                normalize(pattern)
                dataset[ix, :] = pattern
        fa = OnlineFuzzyART(0.95, 0.001, 0.9, reader.num_fields)
        iterations, clusters = fa.run_batch(dataset, max_epochs=100, seed=100)
        self.assertEqual(3, iterations)
        self.assertEqual(52, fa.num_clusters)


if __name__ == '__main__':
    unittest.main()
