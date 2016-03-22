import os
import csv

__author__ = 'Islam Elnabarawy'


class XCSVFileReader(object):
    def __init__(self, filename):
        super(XCSVFileReader, self).__init__()
        self.file = open(filename, 'r')
        # gather some information about the file
        line = self.file.readline()
        self.line_size = len(line)+1
        self.file.seek(0, os.SEEK_END)
        self.file_size = self.file.tell()
        self.num_lines = int(round(self.file_size / self.line_size, 0))
        self.num_fields = len(line.split(','))
        # now reset the file and create a csv.reader
        self.file.seek(0)
        self.reader = csv.reader(self.file)

    def __del__(self):
        self.close()

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, item):
        return self.get_line(item)

    def __next__(self):
        return next(self.reader)

    def get_line(self, line_index):
        self.seek_line(line_index)
        return self.__next__()

    def reset(self):
        self.file.seek(0)

    def seek_line(self, line_index):
        self.file.seek(line_index*self.line_size)

    def close(self):
        self.file.close()
