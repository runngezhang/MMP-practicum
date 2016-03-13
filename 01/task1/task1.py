# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:39:45 2015

@author: timgaripov
"""

import numpy as np


def vectorized(X):
    """
    X --- 2d numpy array
    returns number
    """
    diag = np.diag(X)
    return np.prod(diag[np.nonzero(diag)])


def non_vectorized(X):
    """
    X --- 2d numpy array
    returns number
    """
    n = min(X.shape[0], X.shape[1])
    result = 1
    for i in range(n):
        if X[i, i] != 0:
            result *= X[i, i]
    return result


def alternative(X):
    """
    X --- 2d numpy array
    returns number
    """
    diag = np.diag(X).copy()
    diag[diag == 0] = 1
    return np.prod(diag)


def generate_data(msize=(10, 10)):
    """
    msize --- tuple of 2 integer numbers (default (10, 10))
    returns tuple of a 2d numpy array
    """
    np.random.seed(12345)
    result = np.random.normal(0.0, 1.0, msize)
    n = min(msize[0], msize[1])
    for i in range(n // 3):
        x = np.random.random_integers(0, n - 1)
        result[x, x] = 0
    return (result, )


def make_experiment():
    """
    returns tuplе of 2 lists that describes test cases
    """
    sizes = [(20, 30), (400, 400), (800, 600), (1500, 1500)]
    exp = ([], [])
    for s in sizes:
        exp[0].append(generate_data(s))
        exp[1].append('msize=' + str(s))
    return exp
