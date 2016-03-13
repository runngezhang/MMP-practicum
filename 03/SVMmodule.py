# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:24:08 2015

@author: timgaripov
"""

import numpy as np
import cvxopt
import time
import matplotlib.pyplot as plt
import matplotlib
import sklearn.svm

eps = 1e-6


def generate_data(N=200, D=2, sep=1.0, seed=1535):
    np.random.seed(seed)
    c1 = np.random.uniform(-15.0, 15.0, size=D)
    X1 = np.random.multivariate_normal(c1, np.eye(D, D), size=N)
    r = 4.2
    v = np.random.normal(0.0, 1.0, size=D)
    v /= np.sqrt(np.sum(v**2))
    c2 = c1 + v * 2.0 * r * sep
    X2 = np.random.multivariate_normal(c2, np.eye(D, D), size=N)
    return (np.vstack((X1, X2)), np.array([1] * N + [-1] * N))


def compute_primal_objective(X, y, w0, w, C):
    w = w.ravel()
    y = y.ravel()
    res = 0.5*np.dot(w, w.T)
    res += C*np.sum(np.maximum(0.0, 1.0 - y*(np.dot(w.T, X.T) + w0)))
    return res


def compute_dual_objective(X, y, a, C, gamma=0):
    y = y.ravel()
    a = a.ravel()
    if gamma == 0:
        K = np.dot(X, X.T)
    else:
        s = np.sum(X ** 2, axis=1)
        K = s[:, np.newaxis] - 2.0*np.dot(X, X.T) + s[np.newaxis, :]
        K = np.exp(-gamma*K)
    K *= y[:, np.newaxis] * y[np.newaxis, :]
    K *= a[:, np.newaxis] * a[np.newaxis, :]
    return np.sum(a) - 0.5 * np.sum(np.sum(K, axis=1))


def compute_support_vectors(X, y, A):
    y = y.ravel()
    A = A.ravel()
    ind = np.where(eps < A)[0]
    return X[ind, :]


def compute_w(X, y, A):
    y = y.ravel()
    A = A.ravel()
    w = np.sum(A[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
    return w


def svm_subgradient_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False,
                           part=1.0, alpha=0.9, beta=0.7):
    t1 = time.clock()
    N = X.shape[0]
    K = min(N, max(1, int(N * part)))
    mean = np.mean(X, axis=0)
    X = X.copy() - mean[np.newaxis, :]
    ind = np.arange(N)
    D = X.shape[1]

    np.random.seed(N + D)
    w = np.random.normal(size=D)
    w0 = 0

    cur_obj = compute_primal_objective(X, y, w0, w, C)
    objective_curve = np.array([cur_obj])
    ans = cur_obj
    ans_w = w
    ans_w0 = w0

    status = 1
    for i in range(max_iter):
        if verbose:
            s = 'Iteration #{0}. Objective: {1}'
            print(s.format(i, cur_obj))
        np.random.shuffle(ind)
        CX = X[ind[:K], :]
        cy = y[ind[:K]]
        alpha_t = alpha / ((i + 1) ** beta)
        g = ((1.0 - cy * (np.dot(w, CX.T) + w0)) > -eps)
        dzdw = np.sum(w[np.newaxis, :] / N - C * cy[:, np.newaxis] * CX *
                      g[:, np.newaxis],
                      axis=0)
        dzdw0 = -C * np.sum(cy * g)
        w -= alpha_t * dzdw
        w0 -= alpha_t * dzdw0
        old_obj = cur_obj
        cur_obj = compute_primal_objective(X, y, w0, w, C)
        if ans > cur_obj:
            ans = cur_obj
            ans_w = w
            ans_w0 = w0
        objective_curve = np.append(objective_curve, cur_obj)
        if (np.abs(old_obj - cur_obj) < tol):
            status = 0
            break
    if verbose:
        s = 'Objective: {0}'
        print(s.format(cur_obj))
    t2 = time.clock()
    ans_w0 -= np.dot(ans_w, mean)
    return {'w0': ans_w0,
            'w': ans_w,
            'status': status,
            'objective_curve': objective_curve,
            'time': t2 - t1}


def svm_qp_primal_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False):
    t1 = time.clock()
    N = X.shape[0]
    D = X.shape[1]
    P = np.zeros((1 + D + N, 1 + D + N))
    P[1:D + 1, 1:D + 1] = np.eye(D, D)
    P = cvxopt.matrix(P)
    q = np.zeros((1 + D + N, 1))
    q[1 + D:, 0] = C
    q = cvxopt.matrix(q)

    G = np.hstack((-y[:, np.newaxis],
                   -y[:, np.newaxis] * X,
                   -np.eye(N, N)))
    G = np.vstack((G, np.hstack((np.zeros((N, 1 + D)), -np.eye(N, N)))))
    G = cvxopt.matrix(G)
    h = np.array([-1.0] * N + [0.0] * N)
    h = cvxopt.matrix(h[:, np.newaxis])

    cvxopt.solvers.options['abstol'] = tol
    cvxopt.solvers.options['show_progress'] = verbose
    cvxopt.solvers.options['maxiters'] = max_iter
    res = cvxopt.solvers.qp(P=P, q=q, G=G, h=h)
    t2 = time.clock()
    w = np.array(res['x']).ravel()[1:1+D]
    w0 = res['x'][0]
    return {'status': 0 if res['status'] == 'optimal' else 1,
            'w': w,
            'w0': w0,
            'time': t2 - t1}


def compute_RBF(X, Y, gamma):
    K = np.sum(X ** 2, axis=1)[:, np.newaxis] - \
        2.0*np.dot(X, Y.T) + \
        np.sum(Y ** 2, axis=1)[np.newaxis, :]
    K = np.exp(-gamma*K)
    return K


def svm_qp_dual_solver(X, y, C, tol=1e-6, max_iter=100,
                       verbose=False, gamma=0):
    t1 = time.clock()
    N = X.shape[0]
    # D = X.shape[1]
    if gamma == 0:
        K = np.dot(X, X.T)
    else:
        K = compute_RBF(X, X, gamma)

    y = y.ravel()
    P = K * y[:, np.newaxis] * y[np.newaxis, :]
    P = cvxopt.matrix(P)
    q = -np.ones((N, 1))
    q = cvxopt.matrix(q)

    G = np.vstack((np.eye(N, N), -np.eye(N, N)))
    G = cvxopt.matrix(G)

    h = np.array([1.0 * C] * N + [0.0] * N)
    h = cvxopt.matrix(h[:, np.newaxis])

    E = cvxopt.matrix(1.0 * y[np.newaxis, :])
    b = cvxopt.matrix([0.0])

    cvxopt.solvers.options['abstol'] = tol
    cvxopt.solvers.options['show_progress'] = verbose
    cvxopt.solvers.options['maxiters'] = max_iter
    res = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=E, b=b)
    t2 = time.clock()
    A = np.array(res['x']).ravel()
    w = compute_w(X, y, A)
    w0 = res['y'][0]
    return {'status': 0 if res['status'] == 'optimal' else 1,
            'w': w,
            'w0': w0,
            'A': A,
            'time': t2 - t1}


def svm_liblinear_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False):
    t1 = time.clock()
    clf = sklearn.svm.LinearSVC(C=C, loss='hinge', tol=tol, max_iter=max_iter,
                                verbose=int(verbose), intercept_scaling=4,
                                random_state=14452)
    clf.fit(X, y)
    t2 = time.clock()
    return {'w': clf.coef_.copy(),
            'w0': clf.intercept_[0],
            'time': t2 - t1}


def svm_libsvm_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False,
                      gamma=0):
    t1 = time.clock()
    if gamma == 0:
        clf = sklearn.svm.SVC(C=C, kernel='linear', max_iter=max_iter,
                              verbose=int(verbose), tol=tol,
                              random_state=14452)
    else:
        clf = sklearn.svm.SVC(C=C, kernel='rbf', gamma=gamma,
                              max_iter=max_iter, verbose=int(verbose), tol=tol)
    clf.fit(X, y)
    t2 = time.clock()
    sv = clf.support_
    A = np.zeros(X.shape[0])
    A[sv] = np.abs(clf.dual_coef_.copy())
    w = compute_w(X, y, A)
    return {'w': w,
            'w0': clf.intercept_[0],
            'A': A,
            'time': t2 - t1}


def visualize(X, y, w0, w, A=None, gamma=0):
    if (X.shape[1] != 2) or (w.size != 2):
        raise ValueError('Objects must be 2D')
    pos = np.where(y == 1)[0]
    neg = np.where(y == -1)[0]
    alpha = 0.1
    xrange = [np.min(X[:, 0]), np.max(X[:, 0])]
    xlen = xrange[1] - xrange[0]
    xrange[0] -= xlen * alpha
    xrange[1] += xlen * alpha
    yrange = [np.min(X[:, 1]), np.max(X[:, 1])]
    ylen = yrange[1] - yrange[0]
    yrange[0] -= ylen * alpha
    yrange[1] += ylen * alpha

    prop = ylen / xlen
    plt.figure(figsize=(7, 7 * prop))
    gx = np.linspace(xrange[0], xrange[1], 100)
    gy = np.linspace(yrange[0], yrange[1], 100)
    gx, gy = np.meshgrid(gx, gy)
    Z = np.zeros((100, 100))
    if A is not None:
        SV = compute_support_vectors(X, y, A)
        plt.scatter(SV[:, 0], SV[:, 1], marker='o', s=150, facecolors='none',
                    edgecolors='black', linewidth=1)
    if gamma == 0:
        for i in range(100):
            for j in range(100):
                CX = np.array([gx[i, j], gy[i, j]])
                Z[i, j] = w0 + np.dot(w, CX)
    else:
        for i in range(100):
            for j in range(100):
                CX = np.array([gx[i, j], gy[i, j]])
                K = compute_RBF(CX[np.newaxis, :], X, gamma)
                Z[i, j] = np.sum(A[np.newaxis, :] * y[np.newaxis, :] * K) + w0
    plt.imshow(Z[::-1, :], extent=xrange + yrange, cmap=matplotlib.cm.bwr,
               alpha=0.3)
    plt.colorbar(fraction=0.0455 * prop, pad=0.04,)
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', color='r', s=35, alpha=0.7)
    plt.scatter(X[neg, 0], X[neg, 1], marker='x', color='b', s=35, alpha=0.7)
    plt.contour(gx, gy, Z, colors=['blue', 'black', 'red'], linewidth=0.5,
                levels=[-1.0, 0.0, 1.0], linestyles=['--', '-', '--'])
    plt.xlim(xrange)
    plt.ylim(yrange)

    plt.show()
