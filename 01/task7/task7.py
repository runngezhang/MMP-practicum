# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:18:40 2015

@author: timgaripov
"""

import numpy as np
import scipy.spatial


def scipy_standart(X, Y):
    """
    X, Y --- 2d numpy array with the same shape[1]
    returns --- 2d numpy array with shape (X.shape[0], Y.shape[0])
    """
    return scipy.spatial.distance.cdist(X, Y)


def vectorized(X, Y):
    """
    X, Y --- 2d numpy array with the same shape[1]
    returns --- 2d numpy array with shape (X.shape[0], Y.shape[0])
    """
    return np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2,
                  axis=2) ** 0.5


def alternative(X, Y):
    """
    X, Y --- 2d numpy array with the same shape[1]
    returns --- 2d numpy array with shape (X.shape[0], Y.shape[0])
    """
    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        result[i] = np.sum((X[i]-Y) ** 2, axis=1)
    return np.sqrt(result)


def generate_data(n=10, m=10, d=10):
    """
    n, m, d --- integer numbers (default 10)
    returns tuple of two 2d numpy arrays
    """
    np.random.seed(12345)
    return (np.random.uniform(-1.0, 1.0, size=(n, d)),
            np.random.uniform(-1.0, 1.0, size=(m, d)))


def make_experiment():
    """
    returns tuple of 2 lists that describes test cases
    """
    sizes = [(20, 30, 10), (400, 400, 2),
             (100, 100, 10), (100, 100, 100),
             (100, 100, 500), (100, 100, 1000),
             (400, 400, 20), (400, 400, 100),
             (400, 400, 400), (400, 400, 800)]
    exp = ([], [])
    for s in sizes:
        exp[0].append(generate_data(*s))
        exp[1].append('n=' + str(s[0]) +
                      ', m=' + str(s[1]) +
                      ', d=' + str(s[2]))
    return exp
