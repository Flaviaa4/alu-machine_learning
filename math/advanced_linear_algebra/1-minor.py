#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix."""


def minor(matrix):
    """Calculates the minor matrix of a matrix."""
    if not isinstance(matrix, list) or any(
            not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    for i in range(len(matrix)):
        row = []

        for j in range(len(matrix)):
            submatrix = [
                [matrix[x][y] for y in range(len(matrix))
                 if y != j]
                for x in range(len(matrix))
                if x != i
            ]

            from_0_determinant = __import__('0-determinant')
            determinant = from_0_determinant.determinant
            row.append(determinant(submatrix))

        minor_matrix.append(row)

    return minor_matrix
