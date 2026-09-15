#!/usr/bin/env python3
"""Defines a function that creates the training operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network

    loss: the loss of the network's prediction
    alpha: the learning rate

    Returns: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
