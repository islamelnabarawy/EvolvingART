import argparse
import os

from deap import gp
import pygraphviz as pgv

from main import get_primitive_set


__author__ = 'Islam Elnabarawy'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=['wine', 'iris', 'glass'],
                        help="The name of the dataset to process")
    parser.add_argument("--path", default="ccf/",
                        help="The path for the files to process")
    parser.add_argument("--output", default="output/ccf/",
                        help="The path to write output files to")
    parser.add_argument("--extension", default=".ccf",
                        help="The extension of the file names to read")
    parser.add_argument("--format", default=".pdf",
                        help="The file format to write graphs as")
    args = parser.parse_args()

    process_data(args.dataset, args.extension, args.path, args.output, args.format)


def generate_tree(expr, filename):
    tree = gp.PrimitiveTree.from_string(expr, get_primitive_set())
    nodes, edges, labels = gp.graph(tree)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(filename)


def process_data(dataset, extension, path, output, format):
    print("Dataset: {}".format(dataset))
    print("Path: {}".format(path))
    print("Extension: {}".format(extension))
    print("Output Path: {}".format(output))
    print("File Format: {}".format(format))

    filename = os.path.join(path, '{}{}'.format(dataset, extension))

    with open(filename, 'r') as f:
        for ix, line in enumerate(f.readlines()):
            out_filename = os.path.join(output, '{}-{}{}'.format(dataset, ix, format))
            generate_tree(line, out_filename)


if __name__ == '__main__':
    main()
