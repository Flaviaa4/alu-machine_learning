#!/usr/bin/env python3
"""Module for adding two 2D matrices."""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise."""
    if len(mat1) != len(mat2):
        return None

    if any(len(mat1[i]) != len(mat2[i]) for i in range(len(mat1))):
        return None

    return [[mat1[i][j] + mat2[i][j]
             for j in range(len(mat1[i]))]
            for i in range(len(mat1))]