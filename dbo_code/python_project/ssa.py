"""
SSA – Sparrow Search Algorithm
Translated from SSA.m
"""
import numpy as np
from utils import initialization


def SSA(N, max_iter, lb, ub, dim, fobj):
    ST = 0.7
    PD = 0.3
    SD = 0.5

    PDNumber = round(N * PD)
    SDNumber = round(SD * PDNumber)

    X = initialization(N, dim, ub, lb)
    fitness = np.array([fobj(X[i]) for i in range(N)])

    idx = np.argsort(fitness)
    fitness = fitness[idx]
    X = X[idx]

    GBestF = fitness[0]
    GBestX = X[0].copy()
    X_new = X.copy()

    convergence = np.zeros(max_iter)

    for t in range(max_iter):
        BestF = fitness[0]
        WorstF = fitness[-1]
        R2 = np.random.rand()

        # ── Producers ──
        for j in range(PDNumber):
            if R2 < ST:
                X_new[j] = X[j] * np.exp(-j / (np.random.rand() * max_iter + 1e-30))
            else:
                X_new[j] = X[j] + np.random.randn() * np.ones(dim)

        # ── Scroungers ──
        for j in range(PDNumber, N):
            if j > (N - PDNumber) / 2 + PDNumber:
                X_new[j] = np.random.randn() * np.exp((X[-1] - X[j]) / (j**2 + 1e-30))
            else:
                A = np.where(np.random.rand(dim) > 0.5, 1, -1).astype(float)
                AA = A / (A @ A + 1e-30)
                X_new[j] = X[0] + np.abs(X[j] - X[0]) * AA

        # ── Scouts (danger awareness) ──
        perm = np.random.permutation(N)
        sd_idx = perm[:SDNumber]
        for j in range(SDNumber):
            idx_j = sd_idx[j]
            if fitness[idx_j] > BestF:
                X_new[idx_j] = X[0] + np.random.randn() * np.abs(X[idx_j] - X[0])
            elif fitness[idx_j] == BestF:
                K = 2 * np.random.rand() - 1
                X_new[idx_j] = X[idx_j] + K * (
                    np.abs(X[idx_j] - X[-1])
                    / (fitness[idx_j] - fitness[-1] + 1e-8))

        # Boundary
        X_new = np.clip(X_new, lb, ub)

        # Evaluate
        fitness_new = np.array([fobj(X_new[j]) for j in range(N)])
        for j in range(N):
            if fitness_new[j] < GBestF:
                GBestF = fitness_new[j]
                GBestX = X_new[j].copy()

        X = X_new.copy()
        fitness = fitness_new.copy()
        idx = np.argsort(fitness)
        fitness = fitness[idx]
        X = X[idx]

        convergence[t] = GBestF

    return GBestF, GBestX, convergence
