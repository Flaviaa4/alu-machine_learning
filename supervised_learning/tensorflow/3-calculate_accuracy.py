#!/usr/bin/env python3
"""Defines a function that calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction

    y: a placeholder for the labels of the input data
    y_pred: a tensor containing the network's predictions

    Returns: a tensor containing the decimal accuracy of the prediction
    """
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
