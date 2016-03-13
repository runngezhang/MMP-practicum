# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:15:28 2016

@author: timgaripov
"""

import numpy as np
import math
import PIL


def display_layer(X, filename='layer.png'):
    D = int(math.sqrt(X.shape[1] // 3))
    X = X.copy().reshape(X.shape[0], D, D, 3)
    mx = np.max(X, axis=(0, 1, 2))[np.newaxis,
                                   np.newaxis,
                                   np.newaxis,
                                   :]
    mn = np.min(X, axis=(0, 1, 2))[np.newaxis,
                                   np.newaxis,
                                   np.newaxis,
                                   :]
    X = (X - mn) / (mx - mn)
    P = math.ceil(math.sqrt(X.shape[0]))
    S = P * D + (P - 1) * 2
    img = np.zeros((S, S, 3))
    num = 0
    for i in range(P):
        for j in range(P):
            if (num < X.shape[0]):
                left = j * (D + 2)
                up = i * (D + 2)
                img[up:up+D, left:left+D, :] = X[num]
            num += 1
    #if (D <= 8):
    #    img = scipy.misc.imresize(img, 300, interp='nearest')
    img = PIL.Image.fromarray(np.uint8(img * 255), 'RGB')
    img = img.resize((img.size[0] * 3, img.size[1] * 3))
    img.save(filename)
