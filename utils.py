import operator
from itertools import combinations
from typing import List, Tuple, Any, Hashable, Dict

from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import normalize


def sort_index(lst: ArrayLike, rev=True) -> List[int]:
    """
    gets indexes of lst in order of lst sorting: [2,1,4,3] (rev=True) -> [2,3,0,1]

    :param lst: list to get indexes
    :param rev: true if need reversed sorting
    :return: list of indexes
    """
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s


def n_max_els(lst: ArrayLike, n: int) -> List[int]:
    """
    finds n max elements of lst

    :param lst: list to find max elements
    :param n: number of elements
    :return: list of max elements
    """
    if n <= len(lst):
        return sort_index(lst)[:n]
    return lst


def get_combinations(lst: ArrayLike[Any]) -> List[Tuple[Any]]:
    """
    get all combinations of all possible lengths for an lst

    :param lst: list of elements
    :return: list of combinations (tuples)
    """
    combs = []
    for r in range(1, len(lst) + 1):
        combs.extend(combinations(lst, r))
    return combs


def last_index_of(lst: ArrayLike, value) -> int:
    """
    finds last index of the value in the list

    :param lst: list of elements
    :param value: element to find
    :return: the last index of the element
    """
    return len(lst) - operator.indexOf(list(reversed(lst)), value) - 1


def get_low(column: ArrayLike) -> int:
    """
    find the index of the last '1' in the matrix column (for gaussian elimination)

    :param column: column of matrix
    :return: index of the last 1 in column or -1 if the column contains only 0
    """
    if 1 in column:
        low = last_index_of(column, 1)
    else:
        low = -1
    return low


def get_gauss_form(matrix: NDArray) -> NDArray:
    """
    transform matrix to gauss form

    :param matrix: matrix
    :return: gaussian form of matrix
    """
    matrix = matrix.T
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


def normalize_dict_values(dct: Dict[Hashable, float]) -> Dict[Hashable, float]:
    """
    normalize values of the dict using max norm

    :param dct: dict
    :return: dict with normalized values in range [0, 1]
    """
    keys = list(dct.keys())
    values = normalize([[dct[key] for key in keys]], axis=1, norm="max")[0]

    return {keys[i]: values[i] for i in range(len(keys))}
