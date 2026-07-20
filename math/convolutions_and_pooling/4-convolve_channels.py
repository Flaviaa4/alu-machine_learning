#!/usr/bin/env python3
"""Performs a convolution on images with channels."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels."""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0)
        pw = max((w - 1) * sw + kw - w, 0)
        ph_top = ph // 2
        ph_bottom = ph - ph_top
        pw_left = pw // 2
        pw_right = pw - pw_left

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
        (
            (0, 0),
            (ph_top, ph_bottom),
            (pw_left, pw_right),
            (0, 0)
        ),
        mode='constant'
    )

    oh = (h + ph_top + ph_bottom - kh) // sh + 1
    ow = (w + pw_left + pw_right - kw) // sw + 1

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded[
                    :,
                    i * sh:i * sh + kh,
                    j * sw:j * sw + kw,
                    :
                ] * kernel,
                axis=(1, 2, 3)
            )

    return output
