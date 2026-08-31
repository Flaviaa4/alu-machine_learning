#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network performing binary
    classification"""

    def __init__(self, nx, layers):
        """Class constructor

        nx is the number of input features
        layers is a list representing the number of nodes in each layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for layer in layers:
            if type(layer) is not int or layer < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            layer_size = layers[i]
            prev_size = nx if i == 0 else layers[i - 1]
            self.__weights['W' + str(i + 1)] = (
                np.random.randn(layer_size, prev_size)
                * np.sqrt(2 / prev_size))
            self.__weights['b' + str(i + 1)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases"""
        return self.__weights
