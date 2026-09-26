#!/usr/bin/env python3
"""Defines a function that creates the training operation using gradient
descent with momentum optimization"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network using the
    gradient descent with momentum optimization algorithm

    loss: the loss of the network
    alpha: the learning rate
    beta1: the momentum weight

    Returns: the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
