# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:50:05 2015

@author: timgaripov
"""

import task8
import numpy as np
import numpy.testing as npt
import unittest

dataset = []
answers = []

dataset.append((np.ones((10, 2)), np.zeros(2), np.eye(2)))
answers.append(-np.log(2.0 * np.pi) - np.ones(10))


for i in range(200):
    data = task8.generate_data(50, 20)
    dataset.append(data)
    answers.append(task8.scipy_standart(*data))


class TestFunctions(unittest.TestCase):

        def test_vectorized(self):
            for id, data in enumerate(dataset):
                npt.assert_allclose(task8.vectorized(*data), answers[id],
                                    atol=1e-5, rtol=0)

        def test_alternative(self):
            for id, data in enumerate(dataset):
                npt.assert_allclose(task8.alternative(*data), answers[id],
                                    atol=1e-5, rtol=0)

if __name__ == "__main__":
    unittest.main()
