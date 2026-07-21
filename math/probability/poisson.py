#!/usr/bin/env python3
"""Poisson distribution module."""

import math


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize a Poisson distribution."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculate the probability mass function."""
        k = int(k)
        if k < 0:
            return 0
        return (self.lambtha ** k * math.exp(-self.lambtha)) / math.factorial(k)

    def cdf(self, k):
        """Calculate the cumulative distribution function."""
        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(i) for i in range(k + 1))
