__author__ = 'Islam Elnabarawy'


def chomp(line):
    """
    remove newline characters from line

    :param line: the line to remove the newline characters from
    :type line: str
    :return: the line with any newline characters removed
    :rtype: str
    """
    return line.replace('\n', '')


def split(line, separator=' '):
    """
    Splits line by separator and returns the list of parts

    The line is passed through chomp first to get rid of newline characters

    :param line: the line to split
    :type line: str
    :param separator: the separator character(s), defaults to space
    :type separator: str
    :return: a list of strings that the line splits off into
    :rtype: list
    """
    return chomp(line).split(separator)


def cat_to_bin(values, choices):
    """
    convert a list of categorical field values to a binary occupancy list format

    :param values: the categorical field values
    :type values: list
    :param choices: the category choices to represent
    :type choices: list
    :return: an occupancy list representation of values
    :rtype: list
    """
    result = [0] * len(choices)
    for value in values:
        if value not in choices:
            return None
        result[choices.index(value)] = 1
    return result


def scale_range(x, x_range, y_range=(0.0, 1.0)):
    """
    scale the number x from the range specified by x_range to the range specified by y_range

    :param x: the number to scale
    :type x: float
    :param x_range: the number range that x belongs to
    :type x_range: tuple
    :param y_range: the number range to convert x to, defaults to (0.0, 1.0)
    :type y_range: tuple
    :return: the scaled value
    :rtype: float
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    return (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min
