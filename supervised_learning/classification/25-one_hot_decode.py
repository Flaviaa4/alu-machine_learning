#!/usr/bin/env python3
"""Converts a one-hot matrix into a vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels

    one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None

    return np.argmax(one_hot, axis=0)
