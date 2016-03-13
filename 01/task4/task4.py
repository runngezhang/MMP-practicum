# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 22:18:06 2015

@author: timgaripov
"""

import numpy as np


def vectorized(x):
    """
    x --- 1d numpy array
    returns number
    """
    return np.max(x[np.minimum(np.where(x == 0)[0] + 1, len(x)-1)])


def non_vectorized(x):
    """
    x --- 1d numpy array
    returns number
    """
    result = -np.inf
    for i in range(1, len(x)):
        if x[i - 1] == 0:
            result = max(result, x[i])
    return result


def alternative(x):
    """
    x --- 1d numpy array
    returns number
    """
    result = -np.inf
    x = x[np.minimum(np.where(x == 0)[0] + 1, len(x)-1)]
    for value in x:
        result = max(result, value)
    return result


def generate_data(n=10):
    """
    n --- integer number (default 10)
    returns tuple of one 1d numpy array
    """
    np.random.seed(12345)
    result = np.random.random_integers(-1000, 1000, n)
    result[np.random.random_integers(0, n - 1, n // 2)] = 0
    return (result, )


def make_experiment():
    """
    returns tuple of 2 lists that describes test cases
    """
    sizes = [100000, 1000000, 10000000]
    exp = ([], [])
    for s in sizes:
        exp[0].append(generate_data(s))
        exp[1].append('n=' + str(s))
    return exp
