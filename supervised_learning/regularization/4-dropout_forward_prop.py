#!/usr/bin/env python3
"""Defines a function that conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout

    X: numpy.ndarray of shape (nx, m) containing the input data for the
        network
    weights: a dictionary of the weights and biases of the neural
        network
    L: the number of layers in the network
    keep_prob: the probability that a node will be kept
        All layers except the last use the tanh activation function
        The last layer uses the softmax activation function

    Returns: a dictionary containing the outputs of each layer and the
        dropout mask used on each layer
    """
    cache = {'A0': X}

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.matmul(W, A_prev) + b

        if i == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache
