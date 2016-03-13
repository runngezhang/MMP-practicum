# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:28:55 2016

@author: timgaripov
"""

import numpy as np


def compute_gradient(J, theta):
    eps = 1e-3
    theta_ = theta.copy().ravel()
    grad = np.zeros(theta_.shape)
    for i in range(theta.size):
        theta_[i] -= eps
        f_1 = J(theta_.reshape(theta.shape))
        theta_[i] += 2 * eps
        f_2 = J(theta_.reshape(theta.shape))
        grad[i] = (f_2 - f_1) / (2 * eps)
        theta_[i] -= eps
    return grad.reshape(theta.shape)


def check_gradient():
    tol = 1e-3
    x = np.random.normal(0.0, 1.0, size=(100))
    W = np.random.normal(0.0, 1.0, size=(100, 100))
    J = lambda x: np.sin(np.dot(x.T, np.dot(W, x)))
    grad = compute_gradient(J, x)
    A = np.dot(x.T, np.dot(W, x))
    G = np.cos(A) * np.dot((W + W.T), x)
    err = np.max(np.abs(G - grad) / np.abs(G))
    assert err <= tol, \
           'Relative error is too big. {0:.2g} > {1:.2g}'.format(err, tol)
