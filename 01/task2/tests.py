# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:50:05 2015

@author: timgaripov
"""

import task2
import numpy as np
import numpy.testing as npt
import unittest

dataset = []
answers = []

dataset.append((np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                np.array([0, 1, 0, 2, 3]),
                np.array([1, 1, 2, 2, 0])))
answers.append(np.array([2, 5, 3, 9, 10]))

for i in range(10):
    data = task2.generate_data((100, 100), 50)
    dataset.append(data)
    answers.append(task2.non_vectorized(*data))


class TestFunctions(unittest.TestCase):

        def test_vectorized(self):
            for id, data in enumerate(dataset):
                npt.assert_array_equal(task2.vectorized(*data), answers[id])

        def test_alternative(self):
            for id, data in enumerate(dataset):
                npt.assert_array_equal(task2.alternative(*data), answers[id])

if __name__ == "__main__":
    unittest.main()
