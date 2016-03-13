# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:50:05 2015

@author: timgaripov
"""

import task3
import numpy as np
import numpy.testing as npt
import unittest

dataset = []
answers = []

dataset.append((np.array([1, 5, 3, 5, 5, 2]), np.array([5, 5, 3, 5, 2, 1])))
answers.append(True)
dataset.append((np.array([5, 5, 3, 5, 5, 2]), np.array([5, 5, 3, 5, 2, 1])))
answers.append(False)

for i in range(10):
    data = task3.generate_data(10000)
    dataset.append(data)
    answers.append(task3.non_vectorized(*data))


class TestFunctions(unittest.TestCase):

        def test_vectorized(self):
            for id, data in enumerate(dataset):
                npt.assert_equal(task3.vectorized(*data), answers[id])

        def test_alternative(self):
            for id, data in enumerate(dataset):
                npt.assert_equal(task3.alternative(*data), answers[id])

if __name__ == "__main__":
    unittest.main()
