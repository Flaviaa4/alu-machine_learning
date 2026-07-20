#!/usr/bin/env python3
"""Performs a convolution using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels."""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0)
        pw = max((w - 1) * sw + kw - w, 0)
        ph_top = ph // 2
        ph_bottom = ph - ph_top
        pw_left = pw // 2
        pw_right = pw - pw_left
    elif padding == 'valid':
        ph_top = ph_bottom = pw_left = pw_right = 0
    else:
        ph_top, pw_left = padding
        ph_bottom = ph_top
        pw_right = pw_left

    padded = np.pad(
        images,
        ((0, 0), (ph_top, ph_bottom), (pw_left, pw_right), (0, 0)),
        mode='constant'
    )

    oh = (h + ph_top + ph_bottom - kh) // sh + 1
    ow = (w + pw_left + pw_right - kw) // sw + 1

    output = np.zeros((m, oh, ow, nc))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j, :] = np.sum(
                padded[:, i * sh:i * sh + kh,
                       j * sw:j * sw + kw, :, np.newaxis] * kernels,
                axis=(1, 2, 3)
            )

    return output
