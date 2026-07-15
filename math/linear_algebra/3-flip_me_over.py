#!/usr/bin/env python3
"""Module that transposes a matrix."""


def matrix_transpose(matrix):
    """Returns the transpose of a matrix."""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]