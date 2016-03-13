# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:50:05 2015

@author: timgaripov
"""

import task4
import numpy as np
import numpy.testing as npt
import unittest

dataset = []
answers = []

dataset.append(((np.array([1, 5, 0, 5, 0, 2])), ))
answers.append(5)
dataset.append(((np.array([0, 0, 0, 0, 0, 0])), ))
answers.append(0)

for i in range(10):
    data = task4.generate_data(1000)
    dataset.append(data)
    answers.append(task4.non_vectorized(*data))


class TestFunctions(unittest.TestCase):

        def test_vectorized(self):
            for id, data in enumerate(dataset):
                npt.assert_equal(task4.vectorized(*data), answers[id])

        def test_alternative(self):
            for id, data in enumerate(dataset):
                npt.assert_equal(task4.alternative(*data), answers[id])

if __name__ == "__main__":
    unittest.main()
