# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:50:05 2015

@author: timgaripov
"""

import task5
import numpy as np
import numpy.testing as npt
import unittest

dataset = []
answers = []

dataset.append(((np.array([[[-1, -1, -1], [-1, -1, -1]],
                           [[1, 1, 1], [1, 1, 1]],
                           [[1, 1, 1], [-1, -1, -1]]]),
                 np.array([1, 2, 3]))))
answers.append(np.array([[-6, -6], [6, 6], [6, -6]]))


for i in range(10):
    data = task5.generate_data((70, 70, 3))
    dataset.append(data)
    answers.append(task5.non_vectorized(*data))


class TestFunctions(unittest.TestCase):

        def test_vectorized(self):
            for id, data in enumerate(dataset):
                npt.assert_allclose(task5.vectorized(*data), answers[id])

        def test_alternative(self):
            for id, data in enumerate(dataset):
                npt.assert_allclose(task5.alternative(*data), answers[id])

if __name__ == "__main__":
    unittest.main()
