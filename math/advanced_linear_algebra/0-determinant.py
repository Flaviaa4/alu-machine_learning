#!/usr/bin/env python3
"""Calculates the determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix."""
    if not isinstance(matrix, list) or any(
            not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - \
            matrix[0][1] * matrix[1][0]

    det = 0

    for col in range(len(matrix)):
        submatrix = [
            [matrix[i][j] for j in range(len(matrix))
             if j != col]
            for i in range(1, len(matrix))
        ]

        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(submatrix)

    return det
