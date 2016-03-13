# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:54:11 2015

@author: timgaripov
"""

import time
import numpy as np
import scipy.linalg
import sys


def logpdf(X, mean, cov, diag_cov=False):
    D = mean.size
    cX = X - mean[np.newaxis, :]
    # invC = np.linalg.inv(cv)
    if diag_cov:
        logdet = np.sum(np.log(np.diag(cov)))
    else:
        logdet = np.linalg.slogdet(cov)[1]    
    const = -0.5 * (D * np.log(2.0 * np.pi) + logdet)
    if diag_cov:
        pw = -0.5 * np.sum(cX * ((1.0 / np.diag(cov))[:, np.newaxis] *  cX.T).T, axis=1)
    else:        
        pw = -0.5 * np.sum(cX * scipy.linalg.solve(cov, cX.T).T, axis=1)        
    return const + pw
    # return scipy.stats.multivariate_normal(mean, cv).logpdf(X)


def compute_all(X, K, w, mean, cov, diag_cov=False):
    N = X.shape[0]
    D = X.shape[1]
    logp = np.zeros((N, K))
    gamma = np.zeros((N, K))
    for i in range(K):
        logp[:, i] = logpdf(X, mean[i, :], cov[i, :, :], diag_cov)
    f = logp + np.log(w)[np.newaxis, :]
    mx = np.max(f, axis=1)
    sm = np.sum(np.exp(f - mx[:, np.newaxis]), axis=1)
    gamma = np.exp(f - mx[:, np.newaxis]) / sm[:, np.newaxis]
    mx = np.max(logp, axis=1)
    log_likelihood = np.sum(mx + np.log(np.sum(w[np.newaxis, :] * np.exp(logp - mx[:, np.newaxis]), axis=1)))
    return gamma, log_likelihood


def EM(X, K, tol=1e-4, diag_cov=False, seed=None, verbose=False, reg=1e-3):
    if seed is not None:
        np.random.seed(seed)
    N = X.shape[0]
    D = X.shape[1]
    LB = X.min(axis=0)
    RB = X.max(axis=0)
    w = np.ones(shape=K) / K
    mean = LB + (RB - LB) * np.random.rand(K, D)
    cov = np.tile(np.eye(D), (K, 1, 1))
    old_gamma = None
    gamma = None
    it = 0
    while True:
        # t1 = time.clock()
        it += 1
        old_gamma = gamma
        gamma, log_likelihood = compute_all(X, K, w, mean, cov, diag_cov)
        # print(gamma[:10])
        # print('t1', time.clock() - t1)
        if verbose:
            print('it {0}: {1}'.format(it, log_likelihood))
        if (old_gamma is not None) and np.max(old_gamma - gamma) < tol:
            break
        s = np.sum(gamma, axis=0)
        s = np.maximum(s, 1e-5)
        w = s / N
        mean = np.dot(gamma.T, X) / s[:, np.newaxis]
        if diag_cov:
            for k in range(K):
                d = X - mean[k, :][np.newaxis, :]                
                cov[k, :, :] = np.diag(np.sum(gamma[:, k][:, np.newaxis] * (d**2), axis=0)) / s[k]
        else:
            for k in range(K):
                d = (X - mean[k, :][np.newaxis, :])               
                cov[k, :, :] = np.dot((gamma[:, k][:, np.newaxis] * d).T, d) / s[k]
        cov[:, np.arange(D), np.arange(D)] += reg
        # print(cov)
        # break
        
            
        # print('t2', time.clock() - t1)
    # print(it)
    return w, mean, cov, log_likelihood


def EM_best_run(X, K, R=5, tol=1e-4, diag_cov=False, verbose=False):
    D = X.shape[1]
    w = np.zeros(K)
    mean = np.zeros((K, D))
    cov = np.zeros((K, D, D))
    log_likelihood = -np.inf
    for t in range(R):
        if verbose:
            print('Run', t + 1, end='')
            sys.stdout.flush()
        tic = time.clock()
        cur_w, cur_m, cur_c, cur_l = EM(X, K, tol, diag_cov, verbose=verbose)
        t = time.clock() - tic
        if (cur_l > log_likelihood):
            w = cur_w
            mean = cur_m
            cov = cur_c
            log_likelihood = cur_l
        if verbose:
            print(' done {0:.2f} s.'.format(t))
            sys.stdout.flush()
    return w, mean, cov, log_likelihood


def predict(X, K, w, mean, cov):
    gamma, _ = compute_all(X, K, w, mean, cov)
    return np.argmax(gamma, axis=1)


def generate_params(K, diag_cov=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
    D = 2
    w = np.random.poisson(lam=3.0, size=K).astype('float64') + 1
    w /= np.sum(w)
    mean = 20.0 * np.random.rand(K, D) - 10.0
    cov = np.zeros((K, D, D))
    for i in range(K):
        if diag_cov:
            cov[i, :, :] = np.diag(2.0 * np.random.rand(D) + 0.5)
        else:
            if np.random.rand() > 0.5:
                cov[i, :, :] = np.array([[2, 0.3], [0.3, 0.5]])
            else:
                cov[i, :, :] = np.array([[1.2, -0.7], [-0.7, 3]])
            cov[i, :, :] *= np.random.rand() + 0.1
    return w, mean, cov


def generate_test_data(N, K, w, mean, cov, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.zeros((N, 2))
    s = np.cumsum(w)
    for i in range(N):
        p = np.random.rand()
        ind = 0
        while (s[ind] < p):
            ind += 1
        X[i, :] = np.random.multivariate_normal(mean[ind, :], cov[ind, :, :])
    return X

"""
np.random.seed(544)
N = 10000
D = 1
K = 10
X = np.random.rand(N, D)
print(EM(X, K, seed=11925))
"""