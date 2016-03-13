# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:50:05 2015

@author: timgaripov
"""

import task1
import numpy as np
import numpy.testing as npt
import unittest

dataset = []
answers = []

dataset.append((np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9], [1, 1, 1]]), ))
answers.append(9)

for data in task1.make_experiment()[0]:
    dataset.append(data)
    answers.append(task1.non_vectorized(*data))


class TestFunctions(unittest.TestCase):

        def test_vectorized(self):
            for id, data in enumerate(dataset):
                npt.assert_allclose(task1.vectorized(*data), answers[id])

        def test_alternative(self):
            for id, data in enumerate(dataset):
                npt.assert_allclose(task1.alternative(*data), answers[id])

if __name__ == "__main__":
    unittest.main()
