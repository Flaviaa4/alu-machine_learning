#!/usr/bin/env python3
"""Normal distribution module."""





class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a normal distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = math.sqrt(variance)

    def z_score(self, x):
        """Calculate the z-score of x."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value from a z-score."""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculate the probability density function."""
        coefficient = 1 / (self.stddev * math.sqrt(2 * math.pi))
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        """Calculate the cumulative distribution function."""
        z = (x - self.mean) / (self.stddev * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))
