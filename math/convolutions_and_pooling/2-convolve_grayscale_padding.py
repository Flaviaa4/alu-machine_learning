#!/usr/bin/env python3
"""Performs a convolution with custom padding on grayscale images."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution with custom padding."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output = np.zeros(
        (m, h + 2 * ph - kh + 1, w + 2 * pw - kw + 1)
    )

    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return output
