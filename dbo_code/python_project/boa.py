"""
BOA – Butterfly Optimization Algorithm
Translated from BOA.m
"""
import numpy as np
from utils import initialization


def BOA(N, max_iter, lb, ub, dim, fobj):
    p = 0.8
    power_exponent = 0.1
    sensory_modality = 0.01

    Sol = lb + np.random.rand(N, dim) * (ub - lb)
    Fitness = np.array([fobj(Sol[i]) for i in range(N)])

    best_idx = np.argmin(Fitness)
    fmin = Fitness[best_idx]
    best_pos = Sol[best_idx].copy()
    S = Sol.copy()

    convergence = np.zeros(max_iter)

    for t in range(max_iter):
        for i in range(N):
            Fnew = fobj(S[i])
            FP = sensory_modality * (abs(Fnew) ** power_exponent)

            if np.random.rand() < p:
                dis = np.random.rand() * np.random.rand() * best_pos - Sol[i]
                S[i] = Sol[i] + dis * FP
            else:
                epsilon = np.random.rand()
                JK = np.random.permutation(N)
                dis = epsilon ** 2 * Sol[JK[0]] - Sol[JK[1]]
                S[i] = Sol[i] + dis * FP

            S[i] = np.clip(S[i], lb, ub)
            Fnew = fobj(S[i])

            if Fnew <= Fitness[i]:
                Sol[i] = S[i].copy()
                Fitness[i] = Fnew
            if Fnew <= fmin:
                best_pos = S[i].copy()
                fmin = Fnew

        convergence[t] = fmin
        sensory_modality += 0.025 / (sensory_modality * max_iter + 1e-30)

    return fmin, best_pos, convergence
