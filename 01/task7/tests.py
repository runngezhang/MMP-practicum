# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:50:05 2015

@author: timgaripov
"""

import task7
import numpy as np
import numpy.testing as npt
import unittest

dataset = []
answers = []

dataset.append((np.array([[3, 4], [4, 3]]), np.array([[4, 4], [3, 3]])))
answers.append(np.array([[1, 1], [1, 1]]))

for i in range(10):
    data = task7.generate_data(200, 200, 5)
    dataset.append(data)
    answers.append(task7.scipy_standart(*data))


class TestFunctions(unittest.TestCase):

        def test_vectorized(self):
            for id, data in enumerate(dataset):
                npt.assert_allclose(task7.vectorized(*data), answers[id],
                                    atol=1e-6, rtol=0)

        def test_alternative(self):
            for id, data in enumerate(dataset):
                npt.assert_allclose(task7.alternative(*data), answers[id],
                                    atol=1e-6, rtol=0)

if __name__ == "__main__":
    unittest.main()
