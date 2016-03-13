# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:35:18 2015

@author: timgaripov
"""

import numpy as np
import scipy.stats


def scipy_standart(X, m, C):
    """
    X --- NxD 2d numpy array
    m --- 1d numpy array of size D
    C --- DxD 2d numpy array
    returns 1d numpy array of size N
    """
    return scipy.stats.multivariate_normal(m, C).logpdf(X)


def vectorized(X, m, C):
    """
    X --- NxD 2d numpy array
    m --- 1d numpy array of size D
    C --- DxD 2d numpy array
    returns 1d numpy array of size N
    """
    X = X - m[np.newaxis, :]
    d = len(m)
    invC = np.linalg.inv(C)
    logdet = np.linalg.slogdet(C)[1]
    const = -0.5 * (d * np.log(2.0 * np.pi) + logdet)
    pw = -0.5 * np.diag(np.dot(np.dot(X, invC), X.T))
    return const + pw


def alternative(X, m, C):
    """
    X --- NxD 2d numpy array
    m --- 1d numpy array of size D
    C --- DxD 2d numpy array
    returns 1d numpy array of size N
    """
    X = X - m[np.newaxis, :]
    d = len(m)
    invC = np.linalg.inv(C)
    logdet = np.linalg.slogdet(C)[1]
    const = -0.5 * (d * np.log(2.0 * np.pi) + logdet)
    result = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        result[i] = const - 0.5 * np.dot(np.dot(X[i, :], invC), X[i, :].T)
    return result


def generate_data(n=10, d=2):
    """
    n --- integer number (default 10)
    d --- integer number (default 2)
    return tuple: (2d numpy array, 1d numpy array, 2d numpy array)
    """
    np.random.seed(12345)
    m = np.random.normal(size=d)
    C = np.random.normal(size=(d, d))
    C = np.dot(C.T, C)
    X = np.random.normal(size=(n, d))
    return (X, m, C)


def make_experiment():
    """
    returns tuple of 2 lists that describes test cases
    """
    sizes = [(100, 10), (100, 80), (100, 100), (100, 400),
             (200, 10), (200, 100), (200, 200), (200, 800),
             (2000, 2), (2000, 20), (2000, 100), (2000, 200)]
    exp = ([], [])
    for s in sizes:
        exp[0].append(generate_data(*s))
        exp[1].append('n=' + str(s[0]) + ', d=' + str(s[1]))
    return exp
