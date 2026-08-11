"""
GWO – Grey Wolf Optimizer
Translated from GWO.m
"""
import numpy as np
from utils import initialization


def GWO(N, max_iter, lb, ub, dim, fobj):
    Alpha_pos = np.zeros(dim)
    Alpha_score = np.inf
    Beta_pos = np.zeros(dim)
    Beta_score = np.inf
    Delta_pos = np.zeros(dim)
    Delta_score = np.inf

    Positions = initialization(N, dim, ub, lb)
    convergence = np.zeros(max_iter)

    for l in range(max_iter):
        for i in range(N):
            Positions[i] = np.clip(Positions[i], lb, ub)
            fitness = fobj(Positions[i])

            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i].copy()
            if Alpha_score < fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i].copy()
            if Alpha_score < fitness < Delta_score and fitness > Beta_score:
                Delta_score = fitness
                Delta_pos = Positions[i].copy()

        a = 2 - l * 2 / max_iter

        for i in range(N):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3

        convergence[l] = Alpha_score

    return Alpha_score, Alpha_pos, convergence
