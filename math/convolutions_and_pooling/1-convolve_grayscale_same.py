#!/usr/bin/env python3
"""Performs a same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh - 1
    pw = kw - 1

    ph_top = ph // 2
    ph_bottom = ph - ph_top
    pw_left = pw // 2
    pw_right = pw - pw_left

    padded = np.pad(
        images,
        ((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
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