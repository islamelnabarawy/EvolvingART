import argparse
import os
import re

__author__ = 'Islam Elnabarawy'


def process_file(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            # print(line)
            m = re.search('\s*(.+)\((.*?),\)', line)
            if m:
                data[m.group(2)] = m.group(1)
            else:
                print('Error matching expression on line: {}'.format(line))
    max_perf = max(data.keys())
    return max_perf, data[max_perf]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=['wine', 'iris', 'glass'],
                        help="The name of the dataset to process")
    parser.add_argument("--path", default="output/main/processed/",
                        help="The path for the files to process")
    parser.add_argument("--extension", default=".test",
                        help="The extension of the file names")
    args = parser.parse_args()

    dir_name = os.path.join(args.path, args.dataset)
    files = [i for i in os.listdir(dir_name) if i.endswith(args.extension)]

    for f in files:
        best_perf, best_expr = process_file(os.path.join(dir_name, f))
        print(best_expr)

if __name__ == '__main__':
    main()
