import argparse
import csv
import os

import numpy as np

__author__ = 'Islam Elnabarawy'


def process_file(filename):
    data = {
        'training': {
            'iterations': [],
            'clusters': [],
            'performance': []
        },
        'testing': {
            'iterations': [],
            'clusters': [],
            'performance': []
        }
    }
    with open(filename, 'r') as f:
        for line in f.readlines():
            l = [s.strip() for s in line.split('|') if len(s.strip()) > 0]
            data['training']['iterations'].append(float(l[1]))
            data['training']['clusters'].append(float(l[2]))
            data['training']['performance'].append(float(l[3]))
            data['testing']['iterations'].append(float(l[4]))
            data['testing']['clusters'].append(float(l[5]))
            data['testing']['performance'].append(float(l[6]))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=['wine', 'iris', 'glass'],
                        help="The name of the dataset to process")
    parser.add_argument("--path", default="output/comparative/",
                        help="The path for the files to process")
    parser.add_argument("--extension", default=".out",
                        help="The extension of the file names")
    args = parser.parse_args()

    dir_name = os.path.join(args.path, '{}-avg'.format(args.dataset))
    files = [i for i in os.listdir(dir_name) if i.endswith(args.extension)]

    data = []
    train_results = np.zeros((30, 11), dtype=float)
    test_results = np.zeros((30, 11), dtype=float)
    for ix, f in enumerate(files):
        result = process_file(os.path.join(dir_name, f))
        data.append(result)
        train, test = result['training']['performance'], result['testing']['performance']
        train_results[ix, :] = train
        test_results[ix, :] = test

    write_results(os.path.join(args.path, '{}-train.csv'.format(args.dataset)), train_results)
    write_results(os.path.join(args.path, '{}-test.csv'.format(args.dataset)), test_results)


def write_results(filename, results):
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect=csv.excel, lineterminator='\n')
        writer.writerow(['Run', 'Baseline'] + ['CCF {}'.format(i) for i in range(1, 11)] + ['Winner'])
        for ix, row in enumerate(results):
            writer.writerow([ix+1] + list(row) + [np.argmax(row)])
        results_mean = results.mean(axis=0)
        writer.writerow(['Avg'] + list(results_mean) + [np.argmax(results_mean)])
        results_std = results.std(axis=0)
        writer.writerow(['Stdev'] + list(results_std) + [np.argmax(results_std)])


if __name__ == '__main__':
    main()
