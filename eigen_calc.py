import numpy as np
from numpy.linalg import norm


def householder2(A):
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
    for i in range(100):
        Q, R = householder2(A)
        A = np.dot(R, Q)

    return A, Q, R

if __name__ == '__main__':
    A = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
    A = np.array(A)
    A, Q, R = qr_eig_algorithm(A)

    print(f"Eigenvalues: {np.diagonal(A)}")
    print(np.linalg.eig(A))


