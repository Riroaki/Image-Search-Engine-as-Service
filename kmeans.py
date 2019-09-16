import numpy as np


def clustering(x: np.ndarray, k: int) -> tuple:
    """A fast implementation of k-means clustering algorithm.

    Args:
        x: data to cluster, shape = (n, p).
        k: number of centeroids.

    Returns:
        centeroid vectors, shape = (k, p).
        array of assignment indices, shape = (n,)
    """
    n = x.shape[0]
    ctrs = x[np.random.permutation(n)[:k]]
    idx = np.ones(n)
    x_square = np.expand_dims(np.sum(np.multiply(x, x), axis=1), 1)

    while True:
        distance = -2 * np.matmul(x, ctrs.T)
        distance += x_square
        distance += np.expand_dims(np.sum(ctrs * ctrs, axis=1), 0)
        new_idx = distance.argmin(axis=1)
        if np.array_equal(new_idx, idx):
            break
        idx = new_idx
        ctrs = np.zeros(ctrs.shape)
        for i in range(k):
            ctrs[i] = np.average(x[idx == i], axis=0)

    return ctrs, idx


def assign(x: np.ndarray, ctrs: np.ndarray) -> np.ndarray:
    """Assign each feature in x to nearest centeroid, according to euclidean distance.

    Args:
        x: features to be assigned.
        ctrs: centeroids features.

    Returns:
        array of assignment indices, shape = (n,).
    """
    x_square = np.expand_dims(np.sum(np.multiply(x, x), axis=1), 1)
    distance = -2 * np.matmul(x, ctrs.T)
    distance += x_square
    distance += np.expand_dims(np.sum(ctrs * ctrs, axis=1), 0)
    idx = distance.argmin(axis=1)
    return idx
