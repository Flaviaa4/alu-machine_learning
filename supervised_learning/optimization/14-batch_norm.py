#!/usr/bin/env python3
"""Defines a function that creates a batch normalization layer for a
neural network in tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in
    tensorflow

    prev: the activated output of the previous layer
    n: the number of nodes in the layer to be created
    activation: the activation function that should be used on the
        output of the layer

    Returns: a tensor of the activated output for the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, kernel_initializer=initializer)
    Z = dense(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma,
                                        1e-8)

    if activation is None:
        return Z_norm
    return activation(Z_norm)
