#!/usr/bin/env python3
"""Calculates the inverse of a matrix."""


def inverse(matrix):
    """Calculates the inverse of a matrix."""
    if not isinstance(matrix, list) or any(
            not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    determinant_module = __import__('0-determinant')
    determinant = determinant_module.determinant

    det = determinant(matrix)

    if det == 0:
        return None

    adjugate_module = __import__('3-adjugate')
    adjugate = adjugate_module.adjugate(matrix)

    return [
        [adjugate[i][j] / det for j in range(len(matrix))]
        for i in range(len(matrix))
    ]
