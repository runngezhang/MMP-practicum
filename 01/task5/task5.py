# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 23:32:30 2015

@author: timgaripov
"""

import numpy as np


def vectorized(img, channelsWeights):
    """
    img --- 3d numpy array with shape=(ihHeight, inWidth, numChannels)
    channelsWeights --- 1d numpy array of size=numChannels
    returns 2d numpy array with shape=(inHeight, inWidth)
    """
    return np.sum(img * channelsWeights[np.newaxis, np.newaxis, :], axis=2)


def non_vectorized(img, channelsWeights):
    """
    img --- 3d numpy array with shape=(ihHeight, inWidth, numChannels)
    channelsWeights --- 1d numpy array of size=numChannels
    returns 2d numpy array with shape=(inHeight, inWidth)
    """
    result = np.zeros(img.shape[:2])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                result[i, j] += img[i, j, k] * channelsWeights[k]
    return result


def alternative(img, channelsWeights):
    """
    img --- 3d numpy array with shape=(ihHeight, inWidth, numChannels)
    channelsWeights --- 1d numpy array of size=numChannels
    returns 2d numpy array with shape=(inHeight, inWidth)
    """
    result = np.zeros(img.shape[:2])
    for k in range(img.shape[2]):
        result += img[:, :, k] * channelsWeights[k]
    return result


def generate_data(imsize=(10, 10, 3)):
    """
    imsize --- tuple of 3 integer numbers (default (10, 10, 3))
    returns tuple of a 3d numpy array and a 1d numpy array
    """
    np.random.seed(12345)
    return (np.random.normal(size=imsize),
            np.random.normal(size=imsize[2]))


def make_experiment():
    """
    returns tuple of 2 lists that describes test cases
    """
    sizes = [(100, 100, 5),
             (100, 100, 10),
             (100, 100, 50),
             (100, 100, 100),
             (200, 200, 4),
             (200, 200, 8),
             (200, 200, 12)]
    exp = ([], [])
    for s in sizes:
        exp[0].append(generate_data(s))
        exp[1].append('imsize=' + str(s))
    return exp
