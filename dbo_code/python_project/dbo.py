"""
Original Dung Beetle Optimizer (DBO)
Translated from DBO_old.m
Ref: Jiankai Xue & Bo Shen (2022) J. Supercomputing
"""
import numpy as np
from utils import bounds


def DBO(N, max_iter, lb, ub, dim, fobj):
    P_percent = 0.2
    pNum = round(N * P_percent)

    lb_vec = np.full(dim, lb)
    ub_vec = np.full(dim, ub)

    # Initialization
    x = lb_vec + (ub_vec - lb_vec) * np.random.rand(N, dim)
    fit = np.array([fobj(x[i]) for i in range(N)])

    pFit = fit.copy()
    pX = x.copy()
    XX = pX.copy()

    bestI = np.argmin(fit)
    fMin = fit[bestI]
    bestX = x[bestI].copy()

    convergence = np.zeros(max_iter)

    for t in range(1, max_iter + 1):
        B = np.argmax(fit)
        worse = x[B].copy()
        r2 = np.random.rand()

        # ── Ball-rolling (producers) ──
        for i in range(pNum):
            if r2 < 0.9:
                a = 1 if np.random.rand() > 0.1 else -1
                x[i] = pX[i] + 0.3 * np.abs(pX[i] - worse) + a * 0.1 * XX[i]
            else:
                aaa = np.random.randint(1, 181)
                if aaa in (0, 90, 180):
                    x[i] = pX[i].copy()
                theta = aaa * np.pi / 180
                x[i] = pX[i] + np.tan(theta) * np.abs(pX[i] - XX[i])
            x[i] = bounds(x[i], lb_vec, ub_vec)
            fit[i] = fobj(x[i])

        bestII = np.argmin(fit)
        fMMin = fit[bestII]
        bestXX = x[bestII].copy()

        R = 1 - t / max_iter
        Xnew1 = bounds(bestXX * (1 - R), lb_vec, ub_vec)
        Xnew2 = bounds(bestXX * (1 + R), lb_vec, ub_vec)
        Xnew11 = bounds(bestX * (1 - R), lb_vec, ub_vec)
        Xnew22 = bounds(bestX * (1 + R), lb_vec, ub_vec)

        # ── Brood ball (Eq.4) ──
        for i in range(pNum, min(12, N)):
            x[i] = bestXX + (np.random.rand(dim) * (pX[i] - Xnew1)
                              + np.random.rand(dim) * (pX[i] - Xnew2))
            x[i] = bounds(x[i], Xnew1, Xnew2)
            fit[i] = fobj(x[i])

        # ── Small DBO (Eq.6) ──
        for i in range(min(12, N), min(19, N)):
            x[i] = pX[i] + (np.random.randn() * (pX[i] - Xnew11)
                             + np.random.rand(dim) * (pX[i] - Xnew22))
            x[i] = bounds(x[i], lb_vec, ub_vec)
            fit[i] = fobj(x[i])

        # ── Thief (Eq.7) ──
        for i in range(min(19, N), N):
            x[i] = bestX + np.random.randn(dim) * (np.abs(pX[i] - bestXX)
                                                     + np.abs(pX[i] - bestX)) / 2
            x[i] = bounds(x[i], lb_vec, ub_vec)
            fit[i] = fobj(x[i])

        # Update personal & global best
        XX = pX.copy()
        for i in range(N):
            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i] = x[i].copy()
            if pFit[i] < fMin:
                fMin = pFit[i]
                bestX = pX[i].copy()

        convergence[t - 1] = fMin

    return fMin, bestX, convergence
