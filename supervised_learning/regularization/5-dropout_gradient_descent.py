#!/usr/bin/env python3
"""Defines a function that updates the weights of a neural network with
Dropout regularization using gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout
    regularization using gradient descent

    Y: one-hot numpy.ndarray of shape (classes, m) containing the
        correct labels for the data
    weights: a dictionary of the weights and biases of the neural
        network
    cache: a dictionary of the outputs and dropout masks of each layer
        of the neural network
    alpha: the learning rate
    keep_prob: the probability that a node will be kept
    L: the number of layers of the network
        All layers use the tanh activation function except the last,
        which uses the softmax activation function
        The weights of the network are updated in place
    """
    m = Y.shape[1]
    weights_copy = weights.copy()
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]
        b = weights_copy['b' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA_prev = np.matmul(W.T, dZ)
            dA_prev = (dA_prev * cache['D' + str(i - 1)]) / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
