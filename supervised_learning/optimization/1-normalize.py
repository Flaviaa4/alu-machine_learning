#!/usr/bin/env python3
"""Defines a function that normalizes (standardizes) a matrix"""


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix

    X: numpy.ndarray of shape (d, nx) to normalize
        d is the number of data points
        nx is the number of features
    m: numpy.ndarray of shape (nx,) containing the mean of all features of X
    s: numpy.ndarray of shape (nx,) containing the standard deviation of
        all features of X

    Returns: The normalized X matrix
    """
    return (X - m) / s
