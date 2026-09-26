#!/usr/bin/env python3
"""Defines a class that performs Bayesian optimization on a noiseless
1D Gaussian process"""
import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian
    process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Class constructor

        f: the black-box function to be optimized
        X_init: numpy.ndarray of shape (t, 1) representing the inputs
            already sampled with the black-box function
        Y_init: numpy.ndarray of shape (t, 1) representing the outputs
            of the black-box function for each input in X_init
        bounds: tuple of (min, max) representing the bounds of the
            space in which to look for the optimal point
        ac_samples: the number of samples that should be analyzed
            during acquisition
        l: the length parameter for the kernel
        sigma_f: the standard deviation given to the output of the
            black-box function
        xsi: the exploration-exploitation factor for acquisition
        minimize: a bool determining whether optimization should be
            performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_b, max_b = bounds
        self.X_s = np.linspace(min_b, max_b, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location using the Expected
        Improvement acquisition function

        Returns: X_next, EI
            X_next is a numpy.ndarray of shape (1,) representing the
                next best sample point
            EI is a numpy.ndarray of shape (ac_samples,) containing the
                expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_opt = np.min(self.gp.Y)
            imp = Y_opt - mu - self.xsi
        else:
            Y_opt = np.max(self.gp.Y)
            imp = mu - Y_opt - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
