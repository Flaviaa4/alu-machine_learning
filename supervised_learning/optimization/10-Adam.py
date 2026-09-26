#!/usr/bin/env python3
"""Defines a function that creates the training operation using the Adam
optimization algorithm"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network using the
    Adam optimization algorithm

    loss: the loss of the network
    alpha: the learning rate
    beta1: the weight used for the first moment
    beta2: the weight used for the second moment
    epsilon: a small number to avoid division by zero

    Returns: the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1=beta1, beta2=beta2,
                                        epsilon=epsilon)
    return optimizer.minimize(loss)
