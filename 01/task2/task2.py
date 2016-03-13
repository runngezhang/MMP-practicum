# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:39:45 2015

@author: timgaripov
"""

import numpy as np


def vectorized(X, i, j):
    """
    X --- 2d numpy array.
    i, j --- 1d numpy arrray of the same length
    returns --- 1d numpy array
    """
    return X[i, j]


def non_vectorized(X, i, j):
    """
    X --- 2d numpy array.
    i, j --- 1d numpy arrray of the same length
    returns --- 1d numpy array
    """
    result = []
    for ind in range(len(i)):
        result += [X[i[ind], j[ind]]]
    return np.array(result)


def alternative(X, i, j):
    """
    X --- 2d numpy array
    i, j --- 1d numpy arrray of the same length
    returns --- 1d numpy array
    """
    return np.array([X[i[ind]][j[ind]] for ind in range(len(i))])


def generate_data(msize=(10, 10), n=10):
    """
    msize --- tuple consist of 2 integer numbers  (default (10, 10))
    n --- integer number (default 10)
    returns tuple of a 2d numpy array and
    a two 1d numpy arrays of the same length
    """
    np.random.seed(12345)
    X = np.random.normal(10.0, 5.0, msize)
    i = np.random.random_integers(0, msize[0] - 1, n)
    j = np.random.random_integers(0, msize[1] - 1, n)
    return (X, i, j)


def make_experiment():
    """
    returns tuple of 2 lists that describes test cases
    """
    sizes = [((50, 50), 100), ((500, 750), 1000), ((8000, 4000), 8000)]
    exp = ([], [])
    for s in sizes:
        exp[0].append(generate_data(*s))
        exp[1].append('msize=' + str(s[0]) + ', n=' + str(s[1]))
    return exp
