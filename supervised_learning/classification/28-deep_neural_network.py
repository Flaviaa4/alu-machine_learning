#!/usr/bin/env python3
"""Defines a deep neural network performing multiclass classification"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network performing multiclass
    classification"""

    def __init__(self, nx, layers, activation='sig'):
        """Class constructor

        nx is the number of input features
        layers is a list representing the number of nodes in each layer
        activation is the activation function used in the hidden layers
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
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        """Getter for the hidden layer activation function"""
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        X is a numpy.ndarray with shape (nx, m) that contains the input
        data
        """
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b

            if i == self.__L:
                A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
            elif self.__activation == 'sig':
                A = 1 / (1 + np.exp(-Z))
            else:
                A = np.tanh(Z)

            self.__cache['A' + str(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using cross-entropy

        Y is a one-hot numpy.ndarray with shape (classes, m) that
        contains the correct labels for the input data
        A is a numpy.ndarray with shape (classes, m) containing the
        activated output of the neuron for each example
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        X is a numpy.ndarray with shape (nx, m) that contains the input
        data
        Y is a one-hot numpy.ndarray with shape (classes, m) that
        contains the correct labels for the input data
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A == np.amax(A, axis=0), 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        cache is a dictionary containing all the intermediary values of
        the network
        alpha is the learning rate
        """
        m = Y.shape[1]
        weights = self.__weights.copy()
        dZ = cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = weights['W' + str(i)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if i > 1:
                if self.__activation == 'sig':
                    dA_prev = A_prev * (1 - A_prev)
                else:
                    dA_prev = 1 - A_prev ** 2
                dZ = np.matmul(W.T, dZ) * dA_prev

            self.__weights['W' + str(i)] = W - alpha * dW
            self.__weights['b' + str(i)] = (
                weights['b' + str(i)] - alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network

        X is a numpy.ndarray with shape (nx, m) that contains the input
        data
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        iterations is the number of iterations to train over
        alpha is the learning rate
        verbose defines whether or not to print training progress
        graph defines whether or not to graph training progress
        step is the number of iterations between prints/graph points
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i != iterations:
                self.gradient_descent(Y, cache, alpha)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    steps.append(i)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format

        filename is the file to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object

        filename is the file from which the object should be loaded
        """
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
