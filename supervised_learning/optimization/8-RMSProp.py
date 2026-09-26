#!/usr/bin/env python3
"""Defines a function that creates the training operation using the
RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network using the
    RMSProp optimization algorithm

    loss: the loss of the network
    alpha: the learning rate
    beta2: the RMSProp weight
    epsilon: a small number to avoid division by zero

    Returns: the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                           epsilon=epsilon)
    return optimizer.minimize(loss)
