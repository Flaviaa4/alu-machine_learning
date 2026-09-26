#!/usr/bin/env python3
"""Defines a function that calculates the cost of a neural network with
L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization

    cost: the cost of the network without L2 regularization
    lambtha: the regularization parameter
    weights: a dictionary of the weights and biases (numpy.ndarrays) of
        the neural network
    L: the number of layers in the neural network
    m: the number of data points used

    Returns: the cost of the network accounting for L2 regularization
    """
    norm_sum = 0
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        norm_sum += np.linalg.norm(W) ** 2
    return cost + (lambtha / (2 * m)) * norm_sum
