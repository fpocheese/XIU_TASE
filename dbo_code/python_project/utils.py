"""
Utility functions for population initialization.
"""
import numpy as np


def initialization(n: int, dim: int, ub, lb):
    """Random uniform initialization (mirrors initialization.m)."""
    ub = np.asarray(ub, dtype=float)
    lb = np.asarray(lb, dtype=float)
    if ub.ndim == 0:  # scalar bounds
        return np.random.rand(n, dim) * (ub - lb) + lb
    else:
        positions = np.zeros((n, dim))
        for j in range(dim):
            positions[:, j] = np.random.rand(n) * (ub[j] - lb[j]) + lb[j]
        return positions


def bernoulli_map(n: int, dim: int, ub, lb):
    """Bernoulli chaotic map initialization (mirrors Bernoulli.m)."""
    ub = np.asarray(ub, dtype=float)
    lb = np.asarray(lb, dtype=float)
    X = np.random.rand(n, dim)
    k = 0.484
    for i in range(n):
        for j in range(dim - 1):
            if 0 < X[i, j] <= (1 - k):
                X[i, j + 1] = X[i, j] / (1 - k)
            else:
                X[i, j + 1] = (X[i, j] - 1 + k) / k
    # Map to [lb, ub]
    if ub.ndim == 0:
        X = X * (ub - lb) + lb
    else:
        X = X * (ub - lb) + lb
    return X


def bounds(s, lb, ub):
    """Clip solution to [lb, ub]."""
    return np.clip(s, lb, ub)
