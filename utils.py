import operator
from itertools import combinations
from sklearn.preprocessing import normalize


def sort_index(lst, rev=True):
    """
    gets indexes of lst in order of lst sorting: [2,1,4,3] (rev=True) -> [2,3,0,1]

    :param lst: list to get indexes
    :param rev: true if need reversed sorting
    :return: list of indexes
    """
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s


def n_max_els(lst, n):
    """
    finds n max elements of lst

    :param lst: list to find max elements
    :param n: number of elements
    :return: list of max elements
    """
    if n <= len(lst):
        return sort_index(lst)[:n]
    return lst


def get_combinations(lst):
    combs = []
    for r in range(1, len(lst) + 1):
        combs.extend(combinations(lst, r))
    return combs


def last_index_of(lst, value):
    return len(lst) - operator.indexOf(list(reversed(lst)), value) - 1


def get_low(column):
    if 1 in column:
        low = last_index_of(column, 1)
    else:
        low = -1
    return low


def get_gauss_form(matrix):
    left_low_els = []
    for col in range(len(matrix)):
        column = matrix[col]
        low = get_low(column)
        while low != -1 and low in left_low_els:
            column_to_distract = operator.indexOf(left_low_els, low)
            column = (column - matrix[column_to_distract]) % 2
            low = get_low(column)
        matrix[col] = column
        left_low_els.append(low)
    return matrix.T


def normalize_dict_values(dct):
    keys = dct.keys()
    values = normalize([[dct[key] for key in keys]], axis=1, norm="max")[0]

    return {keys[i]: values[i] for i in range(len(keys))}
