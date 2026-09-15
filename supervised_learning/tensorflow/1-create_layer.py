#!/usr/bin/env python3
"""Defines a function that creates a layer for a neural network"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer for a neural network

    prev: the tensor output of the previous layer
    n: the number of nodes in the layer to create
    activation: the activation function that the layer should use

    Returns: the tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )
    return layer(prev)
