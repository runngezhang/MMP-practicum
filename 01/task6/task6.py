# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:32:43 2015

@author: timgaripov
"""

import numpy as np


def vectorized(x):
    """
    x --- 1d numpy array of integer numbers
    returns tuple of two 1d numpy arrays of integers number
    """
    pos = np.append(np.where(np.diff(x) != 0)[0], len(x)-1)
    return (x[pos], np.diff(np.insert(pos, 0, -1)))


def non_vectorized(x):
    """
    x --- 1d numpy array of integer numbers
    returns tuple of two 1d numpy arrays of integers number
    """
    values = [x[0]]
    cnt = []
    cur = 1
    for i in range(1, len(x)):
        if x[i] != x[i - 1]:
            cnt.append(cur)
            cur = 1
            values.append(x[i])
        else:
            cur += 1
    cnt.append(cur)
    return (np.array(values), np.array(cnt))


def alternative(x):
    """
    x --- 1d numpy array of integer numbers
    returns tuple of two 1d numpy arrays of integers number
    """
    pos = np.where(np.diff(x) != 0)[0]
    if (pos.size == 0):
        return (np.array([x[0]]), np.array(x.size))
    cnt = [pos[0] + 1]
    for i in range(1, len(pos)):
        cnt.append(pos[i]-pos[i-1])
    cnt.append(len(x) - pos[-1] - 1)
    values = x[pos]
    values = np.append(values, x[-1])
    return (values, np.array(cnt))


def generate_data(n=10):
    """
    n --- integer number (default 10)
    return tuple of a 1d numpy array
    """
    np.random.seed(12345)
    x = np.random.random_integers(-1000000, 1000000, n)
    for i in range(min(100, n)):
        pos = np.random.random_integers(0, n - 1)
        length = min(n - pos, np.random.random_integers(0, min(100, n)))
        x[pos:pos+length] = x[pos]
    return (x, )


def make_experiment():
    """
    returns tuple of 2 lists that describes test cases
    """
    sizes = [10000, 100000, 1000000]
    exp = ([], [])
    for s in sizes:
        exp[0].append(generate_data(s))
        exp[1].append('n=' + str(s))
    return exp
