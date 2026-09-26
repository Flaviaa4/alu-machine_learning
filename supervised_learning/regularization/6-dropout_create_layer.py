#!/usr/bin/env python3
"""Defines a function that creates a layer of a neural network using
dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout

    prev: a tensor containing the output of the previous layer
    n: the number of nodes the new layer should contain
    activation: the activation function that should be used on the
        layer
    keep_prob: the probability that a node will be kept

    Returns: the output of the new layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=initializer
    )
    dropout = tf.layers.Dropout(rate=1 - keep_prob)
    return dropout(layer(prev))
