# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:48:06 2016

@author: timgaripov
"""

import math
import numpy as np


def normalize_data(images):
    mean = np.mean(images, axis=0)[np.newaxis, :]
    std = np.std(images, axis=0)[np.newaxis, :]
    images = np.minimum(images, mean + 3.0 * std)
    images = np.maximum(images, mean - 3.0 * std)
    images = 0.1 + 0.8 * (images - (mean - 3.0 * std)) / (6.0 * std + 1e-3)
    return images


def sample_patches_raw(images, num_patches=10000, patch_size=8):
    D = int(math.sqrt(images.shape[1] // 3))
    ind = np.random.randint(0, images.shape[0], num_patches)

    left = np.random.randint(0, D - patch_size + 1, num_patches)
    row_num = np.repeat(np.arange(patch_size), patch_size).reshape(patch_size,
                                                                   patch_size)
    row_num = left[:, np.newaxis, np.newaxis] + row_num[np.newaxis, :, :]

    up = np.random.randint(0, D - patch_size + 1, num_patches)
    col_num = np.tile(np.arange(patch_size), patch_size).reshape(patch_size,
                                                                 patch_size)
    col_num = up[:, np.newaxis, np.newaxis] + col_num[np.newaxis, :, :]
    img = images.reshape(images.shape[0], D, D, 3)
    pathces = img[ind[:, np.newaxis, np.newaxis], row_num, col_num, :]

    return pathces.reshape(num_patches, 3 * patch_size ** 2)


def sample_patches(images, num_patches=10000, patch_size=8):
    return normalize_data(sample_patches_raw(images,
                                             num_patches,
                                             patch_size))
