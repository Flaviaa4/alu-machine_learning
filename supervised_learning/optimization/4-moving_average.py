#!/usr/bin/env python3
"""Defines a function that calculates the weighted moving average"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set

    data: the list of data to calculate the moving average of
    beta: the weight used for the moving average
        Uses bias correction

    Returns: a list containing the moving averages of data
    """
    averages = []
    v = 0
    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        averages.append(v / (1 - beta ** (i + 1)))
    return averages
