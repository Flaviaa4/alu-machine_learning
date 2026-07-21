#!/usr/bin/env python3
"""Calculates the sum of squares."""


def summation_i_squared(n):
    """Return the sum of i squared from 1 to n."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
