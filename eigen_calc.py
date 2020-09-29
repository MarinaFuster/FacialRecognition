import numpy as np
from numpy.linalg import norm


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


