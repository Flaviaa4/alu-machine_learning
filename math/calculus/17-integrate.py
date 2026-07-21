#!/usr/bin/env python3
"""Calculates the integral of a polynomial."""


def poly_integral(poly, C=0):
    """Return the integral of a polynomial."""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not isinstance(C, int) or isinstance(C, bool):
        return None

    for coefficient in poly:
        if not isinstance(coefficient, (int, float)):
            return None

    integral = [C]

    for i, coefficient in enumerate(poly):
        value = coefficient / (i + 1)

        if value.is_integer():
            value = int(value)

        integral.append(value)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
