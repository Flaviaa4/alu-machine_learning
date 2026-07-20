#!/usr/bin/env python3
"""Performs a same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        images,
        ((0, 0), (ph, ph - 1), (pw, pw - 1)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return output