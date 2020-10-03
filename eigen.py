from typing import Union

import numpy as np
from numpy.linalg import norm

DELTA = 10 ** -8
MAX_ITERATIONS = 1000


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


def found_eigenvalues(original_matrix, iterated_matrix):
    maybe_eigenvalues = iterated_matrix.diagonal()
    for value in maybe_eigenvalues:
        if not np.linalg.det(original_matrix - np.eye(len(maybe_eigenvalues)) * value) < DELTA:
            return False

    return True


def eigenvectors_stabilized(new_vec, curr_vec):
    return np.linalg.norm(np.subtract(new_vec, curr_vec)) < DELTA


def qr_eig_algorithm(A):
    """
    Calculate Eigenvalues and Eigenvectors using QR algorithm
    See: https://www.physicsforums.com/threads/how-do-i-numerically-find-eigenvectors-for-given-eigenvalues.561763/
    """
    a = A.copy()
    eig_vec = np.identity(a.shape[0])
    for i in range(MAX_ITERATIONS):
        Q, R = householder(a)
        a = R.dot(Q)
        new_eig = eig_vec.dot(Q)
        if found_eigenvalues(A, a) and eigenvectors_stabilized(new_eig, eig_vec):
            break
        eig_vec = new_eig

    eig_val = np.diag(a)

    sort = np.argsort(np.absolute(eig_val))[::-1]
    # We return the transpose because we want each row to give an eigenvector (instead of each column)
    return eig_val[sort], eig_vec[sort]
