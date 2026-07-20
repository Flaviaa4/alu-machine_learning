#!/usr/bin/env python3
"""Performs a convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
        ph_top = ph
        ph_bottom = ph
        pw_left = pw
        pw_right = pw

    elif padding == 'valid':
        ph_top = 0
        ph_bottom = 0
        pw_left = 0
        pw_right = 0

    else:
        ph_top = padding[0]
        ph_bottom = padding[0]
        pw_left = padding[1]
        pw_right = padding[1]

    padded = np.pad(
        images,
        ((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant'
    )

    oh = (h + ph_top + ph_bottom - kh) // sh + 1
    ow = (w + pw_left + pw_right - kw) // sw + 1

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded[:, i * sh:i * sh + kh,
                       j * sw:j * sw + kw] * kernel,
                axis=(1, 2)
            )

    return output
