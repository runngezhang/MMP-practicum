# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:50:05 2015

@author: timgaripov
"""

import task6
import numpy as np
import numpy.testing as npt
import unittest

dataset = []
answers = []

dataset.append(((np.array([1, 5, 5, 0, 0, 2, 2, 2, 3])), ))
answers.append((np.array([1, 5, 0, 2, 3]), np.array([1, 2, 2, 3, 1])))

dataset.append(((np.array([10])), ))
answers.append(((np.array([10]), np.array([1]))))

dataset.append(((np.array([0, 0, 0, 0, 0, 1])), ))
answers.append(((np.array([0, 1]), np.array([5, 1]))))

dataset.append(((np.array([0, 0, 0, 0, 0, 0])), ))
answers.append(((np.array([0]), np.array([6]))))

for i in range(10):
    data = task6.generate_data(1000)
    dataset.append(data)
    answers.append(task6.non_vectorized(*data))


class TestFunctions(unittest.TestCase):

        def test_vectorized(self):
            for id, data in enumerate(dataset):
                npt.assert_equal(task6.vectorized(*data), answers[id])

        def test_alternative(self):
            for id, data in enumerate(dataset):
                npt.assert_equal(task6.alternative(*data), answers[id])

if __name__ == "__main__":
    unittest.main()
