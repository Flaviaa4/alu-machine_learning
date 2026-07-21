#!/usr/bin/env python3
"""Binomial distribution module."""



class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a binomial distribution."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p_estimate = 1 - (variance / mean)
            n_estimate = round(mean / p_estimate)

            self.n = int(n_estimate)
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculate the probability mass function."""
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        coefficient = math.factorial(self.n) / (
            math.factorial(k) * math.factorial(self.n - k)
        )

        return coefficient * (self.p ** k) * (
            (1 - self.p) ** (self.n - k)
        )

    def cdf(self, k):
        """Calculate the cumulative distribution function."""
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        return sum(self.pmf(i) for i in range(k + 1))
