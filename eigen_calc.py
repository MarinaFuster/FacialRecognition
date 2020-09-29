from math import sqrt
import numpy as np
from numpy.linalg import norm


def mult_matrix(M, N):
    """Multiply square matrices of same dimension M and N"""
    # Converts N into a list of tuples of columns
    tuple_N = zip(*N)

    # Nested list comprehension to calculate matrix multiplication
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]


def Q_i(Q_min, i, j, k):
    """Construct the Q_t matrix by left-top padding the matrix Q
    with elements from the identity matrix."""
    if i < k or j < k:
        return float(i == j)
    else:
        return Q_min[i-k][j-k]


def householder(A):
    n = len(A)

    # Set R equal to A, and create Q as a zero matrix of the same size
    R = np.copy(A)
    Q = np.identity(n)

    # The Householder procedure
    for k in range(n):  # We don't perform the procedure on a 1x1 matrix, so we reduce the index by 1
        v = R[k:, k]  # kth column
        ul = v[0] + np.sign(v[0])*norm(v)
        u = v/ul
        u[0] = 1
        beta = np.sign(v[0]) * ul / norm(v)
        R[k:, :] = R[k:, :] - beta * np.outer(u, u).dot(R[k:])
        Q[:, k:] = Q[:, k:] - beta * Q[:, k:].dot(np.outer(u, u))

    return Q.T, R

if __name__ == '__main__':
    A = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
    A = np.array(A)
    Q, R = householder(A)
    print(f"Q: {Q}")
    print(f"R: {R}")


