#!/usr/bin/env python3
"""Calculates the cofactor matrix of a matrix."""


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix."""
    if not isinstance(matrix, list) or any(
            not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_module = __import__('1-minor')
    minor_matrix = minor_module.minor(matrix)

    cofactor_matrix = []

    for i in range(len(matrix)):
        row = []

        for j in range(len(matrix)):
            row.append(minor_matrix[i][j] * (-1) ** (i + j))

        cofactor_matrix.append(row)

    return cofactor_matrix
