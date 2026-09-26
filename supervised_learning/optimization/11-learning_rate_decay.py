#!/usr/bin/env python3
"""Defines a function that updates the learning rate using inverse time
decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy

    alpha: the original learning rate
    decay_rate: the weight used to determine the rate at which alpha
        will decay
    global_step: the number of passes of gradient descent that have
        elapsed
    decay_step: the number of passes of gradient descent that should
        occur before alpha is decayed further
        the learning rate decay occurs in a stepwise fashion

    Returns: the updated value for alpha
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
