
import numpy as np


def vectorize_distance(a, b):
    """
    Calculate euclid distance on each row of a and b
    :param a: Nx... np.array
    :param b: Mx... np.array
    :return: MxN np.array representing correspond distance
    """
    N = a.shape[0]
    a = a.reshape ( N, -1 )
    M = b.shape[0]
    b = b.reshape ( M, -1 )
    a2 = np.tile ( np.sum ( a ** 2, axis=1 ).reshape ( -1, 1 ), (1, M) )
    b2 = np.tile ( np.sum ( b ** 2, axis=1 ), (N, 1) )
    dist = a2 + b2 - 2 * (a @ b.T)
    return np.sqrt ( dist )
