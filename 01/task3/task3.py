# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:39:45 2015

@author: timgaripov
"""

import numpy as np


def vectorized(x, y):
    """
    x, y --- 1d numpy array of integer numbers
    returns boolean
    """
    return np.all(np.sort(x) == np.sort(y))


def non_vectorized(x, y):
    """
    x, y --- 1d numpy array of integer numbers
    returns boolean
    """
    return sorted(list(x)) == sorted(list(y))


def alternative(x, y):
    """
    x, y --- 1d numpy arrays of integer numbers
    returns boolean
    """
    x_vals, x_cnt = np.unique(x, return_counts=True)
    y_vals, y_cnt = np.unique(y, return_counts=True)
    return np.all(x_vals == y_vals) and np.all(x_cnt == y_cnt)


def generate_data(n=10):
    """
    n --- integer number (default 10)
    returns tuple of two 1d numpy arrays of integer numbers
    """
    np.random.seed(12345)
    left = n/2
    right = (n+1)/2
    x = np.hstack((np.random.random_integers(1, 5, left),
                   np.random.random_integers(1, 10000000, right)))
    y = np.hstack((np.random.random_integers(1, 5, left),
                   np.random.random_integers(1, 10000000, right)))
    np.random.shuffle(x)
    np.random.shuffle(y)
    return (x, y)


def make_experiment():
    """
    returns tuple of 2 lists that describes test cases
    """
    sizes = [10000, 100000, 1000000, 10000000]
    exp = ([], [])
    for s in sizes:
        exp[0].append(generate_data(s))
        exp[1].append('n=' + str(s))
    return exp
