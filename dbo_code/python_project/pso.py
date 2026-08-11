"""
PSO – Particle Swarm Optimization
Translated from PSO.m
"""
import numpy as np


def PSO(N, max_iter, lb, ub, dim, fobj):
    Vmax = (ub - lb) * 0.15
    w = 0.8
    c1 = 1.5
    c2 = 1.5

    vel = np.zeros((N, dim))
    pBestScore = np.full(N, np.inf)
    pBest = np.zeros((N, dim))
    gBest = np.zeros(dim)
    gBestScore = np.inf

    pos = lb + np.random.rand(N, dim) * (ub - lb)
    convergence = np.zeros(max_iter)

    for l in range(max_iter):
        # Clip to bounds
        pos = np.clip(pos, lb, ub)

        for i in range(N):
            fitness = fobj(pos[i])
            if fitness < pBestScore[i]:
                pBestScore[i] = fitness
                pBest[i] = pos[i].copy()
            if fitness < gBestScore:
                gBestScore = fitness
                gBest = pos[i].copy()

        # Update velocity & position
        for i in range(N):
            for j in range(dim):
                vel[i, j] = (w * vel[i, j]
                             + c1 * np.random.rand() * (pBest[i, j] - pos[i, j])
                             + c2 * np.random.rand() * (gBest[j] - pos[i, j]))
                vel[i, j] = np.clip(vel[i, j], -Vmax, Vmax)
                pos[i, j] += vel[i, j]

        convergence[l] = gBestScore

    return gBestScore, gBest, convergence
