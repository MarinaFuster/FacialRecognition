from typing import Union

import numpy as np
from numpy.linalg import norm


def householder(A):
    """ Householder method to decompose A into its QR form"""
    m = A.shape[0]
    n = A.shape[1]

    Q = np.eye(m)
    R = np.copy(A)

    for j in range(n):
        x = A[j:, j]
        k = x.shape[0]

        s = -np.sign(x[0]) * norm(x, 2)
        e = np.zeros(k)
        e[0] = 1
        v = (1 / (x[0] - s)) * (x - (s * e))

        R[j:, :] = R[j:, :] - (2 / (v @ v)) * ((np.outer(v, v)) @ R[j:])
        Q[j:] = Q[j:] - (2 / (v @ v)) * ((np.outer(v, v)) @ Q[j:])

    return Q.T, R


def qr_eig_algorithm(A):
    """
    Calculate Eigenvalues and Eigenvectors using QR algorithm
    See: https://www.physicsforums.com/threads/how-do-i-numerically-find-eigenvectors-for-given-eigenvalues.561763/
    """
    Q, R = householder(A)
    eig_vec = Q
    A = np.matmul(R, Q)
    for i in range(100):
        Q, R = householder(A)
        A = np.matmul(R, Q)
        eig_vec = np.matmul(eig_vec, Q)

    eig_val = np.diagonal(A)

    # We return the transpose because we want each row to give an eigenvector (instead of each column)
    return eig_val[::-1], eig_vec.T[::-1]


if __name__ == '__main__':
    A = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
    A = np.array(A)
    eig_val, eig_vec = qr_eig_algorithm(np.copy(A))
    print(f"Eigenvalues: {eig_val}")
    print(f"Eigenvectors: {eig_vec}")
    print(np.linalg.eig(A))


