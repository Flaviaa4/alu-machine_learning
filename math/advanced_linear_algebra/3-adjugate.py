#!/usr/bin/env python3
"""Calculates the adjugate matrix of a matrix."""


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix."""
    if not isinstance(matrix, list) or any(
            not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_module = __import__('2-cofactor')
    cofactor_matrix = cofactor_module.cofactor(matrix)

    return [
        [cofactor_matrix[j][i] for j in range(len(matrix))]
        for i in range(len(matrix))
    ]
