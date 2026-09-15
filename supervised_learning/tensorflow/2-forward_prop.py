#!/usr/bin/env python3
"""Defines a function that creates the forward propagation graph"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network

    x: the placeholder for the input data
    layer_sizes: a list containing the number of nodes in each layer
    activations: a list containing the activation functions for each layer

    Returns: the prediction of the network in tensor form
    """
    prediction = x
    for n, activation in zip(layer_sizes, activations):
        prediction = create_layer(prediction, n, activation)
    return prediction
