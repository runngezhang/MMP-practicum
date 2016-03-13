# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:46:40 2016

@author: timgaripov
"""

import numpy as np
import math


def initialize(hidden_size, visible_size):
    sizes = np.concatenate(([visible_size], hidden_size, [visible_size]))
    theta = np.array([], dtype=np.float64)
    for i in range(sizes.size - 1):
        n_in = sizes[i]
        n_out = sizes[i + 1]
        r = math.sqrt(6 / (n_in + n_out + 1))
        W = np.random.uniform(-r, r, (n_in, n_out)).astype(np.float64)
        b = np.zeros(n_out, dtype=np.float64)
        theta = np.append(theta, W.ravel())
        theta = np.append(theta, b.ravel())
    return theta

def get_params(theta, visible_size, hidden_size):
    W = []
    b = []
    sizes = np.concatenate(([visible_size], hidden_size, [visible_size]))
    pos = 0
    for i in range(sizes.size - 1):
        n_in = sizes[i]
        n_out = sizes[i + 1]
        W.append(theta[pos:pos + n_in*n_out].reshape(n_in, n_out))
        pos += n_in * n_out
        b.append(theta[pos:pos + n_out])
        pos += n_out
    return W, b


def autoencoder_loss(theta,
                     visible_size,
                     hidden_size,
                     lambda_,
                     sparsity_param,
                     beta,
                     data):
    N = data.shape[0]
    sizes = np.concatenate(([visible_size], hidden_size, [visible_size]))
    W, b = get_params(theta, visible_size, hidden_size)
    layers_num = len(W)
    LIN = [None] * layers_num
    SIG = [None] * layers_num
    RHO = [None] * layers_num
    X = data
    LOSS_2 = 0.0
    LOSS_3 = 0.0
    rho = sparsity_param
    for i in range(layers_num):
        LIN[i] = np.dot(X, W[i]) + b[i][np.newaxis, :]
        SIG[i] = 1.0 / (1.0 + np.exp(-LIN[i]))
        RHO[i] = np.mean(SIG[i], axis=0)
        LOSS_2 += np.sum(W[i] ** 2)
        if (i == layers_num // 2 - 1):
            LOSS_3 += np.sum(rho * np.log(rho / RHO[i]) +
                            (1.0 - rho) * np.log((1.0 - rho) / (1.0 - RHO[i])))
        X = SIG[i]
    LOSS_1 = np.sum((SIG[-1] - data) ** 2) / (2.0 * N)
    LOSS_2 *= 0.5 * lambda_
    LOSS_3 *= beta
    LOSS = LOSS_1 + LOSS_2 + LOSS_3

    DLDW = []
    DLDb = []
    # LOSS2 GRADIENTS
    for i in range(layers_num):
        DLDW.append(lambda_ * W[i])
        DLDb.append(np.zeros(b[i].shape, dtype=np.float64))

    # LOSS1 AND LOSS3 GRADIENTS
    DLDSIG = [None] * layers_num
    DLDRHO = [None] * layers_num
    DLDLIN = [None] * layers_num
    grad = np.zeros(theta.shape, dtype=np.float64)
    pos = theta.size
    for i in range(layers_num - 1, -1, -1):
        DLDSIG[i] = np.zeros(SIG[i].shape)
        if (i == layers_num // 2 - 1):
            DLDRHO[i] = beta * (-rho / RHO[i] + (1.0 - rho) / (1.0 - RHO[i]))
            DLDSIG[i] = np.tile(DLDRHO[i] / N, (SIG[i].shape[0], 1))
        if (i == layers_num - 1):
            DLDSIG[i] += (SIG[-1] - data) / N
        else:
            DLDSIG[i] += np.dot(DLDLIN[i + 1], W[i + 1].T)
        DLDLIN[i] = -SIG[i] * (SIG[i] - 1.0) * DLDSIG[i]
        if (i == 0):
            X = data
        else:
            X = SIG[i - 1]
        DLDW[i] += np.dot(X.T, DLDLIN[i])
        DLDb[i] += np.sum(DLDLIN[i], axis=0)
        n_in = sizes[i]
        n_out = sizes[i + 1]
        pos -= n_out
        grad[pos: pos + n_out] = DLDb[i].ravel()
        pos -= n_in * n_out
        grad[pos: pos + n_in * n_out] = DLDW[i].ravel()
    return float(LOSS), grad


def autoencoder_transform(theta,
                          visible_size,
                          hidden_size,
                          layer_num,
                          data):
    Y = data
    W, b = get_params(theta, visible_size, hidden_size)
    for i in range(layer_num):
        Y = np.dot(Y, W[i]) + b[i][np.newaxis, :]
        Y = 1.0 / (1.0 + np.exp(-Y))
    return Y
