# """
# IDBO – Improved Dung Beetle Optimizer for Discrete Task Assignment
#   1. Bernoulli chaotic map initialization
#   2. Variable spiral strategy (brood-ball phase)
#   3. Lévy flight (thief phase)
#   4. Nonlinear weight (thief phase)
#   5. Discrete rounding repair after each position update
#   6. Greedy neighbourhood search on global best (per-dimension swap)
#   7. Opposition-based learning for population diversity

# Adapted from BCLDBO for discrete UAV task assignment.
# """
# import numpy as np
# from math import gamma as math_gamma
# from utils import initialization, bernoulli_map, bounds


# def _discrete_repair(x, lb_vec, ub_vec):
#     """Round to nearest integer and clip to bounds – key for discrete problems."""
#     return np.clip(np.round(x), lb_vec, ub_vec)


# def _greedy_neighbourhood(bestX, fMin, dim, lb_val, ub_val, fobj):
#     """For each dimension, try all integer values and keep the best.
#     This is a lightweight local search (dim × n_uav evaluations)."""
#     improved = bestX.copy()
#     improved_fit = fMin
#     for d in range(dim):
#         original = improved[d]
#         for v in range(int(lb_val), int(ub_val) + 1):
#             if v == original:
#                 continue
#             candidate = improved.copy()
#             candidate[d] = float(v)
#             f_cand = fobj(candidate)
#             if f_cand < improved_fit:
#                 improved_fit = f_cand
#                 improved = candidate.copy()
#     return improved, improved_fit


# def IDBO(N, max_iter, lb, ub, dim, fobj):
#     P_percent = 0.2
#     pNum = round(N * P_percent)

#     lb_val = float(lb)
#     ub_val = float(ub)
#     lb_vec = np.full(dim, lb_val)
#     ub_vec = np.full(dim, ub_val)

#     # ── Bernoulli chaotic map + standard initialization ──
#     x = initialization(N, dim, ub, lb)
#     x = _discrete_repair(x, lb_vec, ub_vec)
#     XSi = bernoulli_map(N, dim, ub, lb)
#     XSi = _discrete_repair(XSi, lb_vec, ub_vec)

#     fit = np.array([fobj(x[i]) for i in range(N)])

#     pFit = fit.copy()
#     pX = x.copy()
#     XX = pX.copy()

#     bestI = np.argmin(fit)
#     fMin = fit[bestI]
#     bestX = x[bestI].copy()

#     convergence = np.zeros(max_iter)

#     # Greedy search interval (not every iteration to save cost)
#     greedy_interval = max(1, max_iter // 20)

#     for t in range(1, max_iter + 1):
#         B = np.argmax(fit)
#         worse = x[B].copy()
#         r2 = np.random.rand()

#         # ── Ball-rolling (producers) ──
#         for i in range(pNum):
#             if r2 < 0.9:
#                 a_val = 1 if np.random.rand() > 0.1 else -1
#                 b_val = np.random.rand()
#                 x[i] = pX[i] + b_val * np.abs(pX[i] - worse) + a_val * 0.1 * XX[i]
#             else:
#                 aaa = np.random.randint(1, 181)
#                 if aaa in (0, 90, 180):
#                     x[i] = pX[i].copy()
#                 theta = aaa * np.pi / 180
#                 x[i] = pX[i] + np.tan(theta) * np.abs(pX[i] - XX[i])
#             x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
#             fit[i] = fobj(x[i])

#         bestII = np.argmin(fit)
#         bestXX = x[bestII].copy()

#         R = 1 - t / max_iter
#         Xnew1 = _discrete_repair(bestXX * (1 - R), lb_vec, ub_vec)
#         Xnew2 = _discrete_repair(bestXX * (1 + R), lb_vec, ub_vec)
#         Xnew11 = _discrete_repair(bestX * (1 - R), lb_vec, ub_vec)
#         Xnew22 = _discrete_repair(bestX * (1 + R), lb_vec, ub_vec)

#         # ── Brood ball with Variable Spiral Strategy ──
#         k_spiral = 5
#         l_spiral = 2 * np.random.rand() - 1
#         z_spiral = np.exp(k_spiral * np.cos(np.pi * (t / max_iter)))
#         spiral_factor = np.exp(z_spiral * l_spiral) * np.cos(2 * np.pi * l_spiral)

#         for i in range(pNum, min(12, N)):
#             x[i] = bestXX + (spiral_factor * np.random.rand(dim) * (pX[i] - Xnew1)
#                               + spiral_factor * np.random.rand(dim) * (pX[i] - Xnew2))
#             x[i] = _discrete_repair(x[i], Xnew1, Xnew2)
#             fit[i] = fobj(x[i])

#         # ── Small DBO with Spiral Strategy ──
#         k_spiral = 5
#         l_spiral = 2 * np.random.rand() - 1
#         z_spiral = np.exp(k_spiral * np.cos(np.pi * (t / max_iter)))
#         spiral_factor = np.exp(z_spiral * l_spiral) * np.cos(2 * np.pi * l_spiral)

#         for i in range(min(12, N), min(19, N)):
#             x[i] = (spiral_factor * pX[i]
#                      + np.random.randn() * (pX[i] - Xnew11)
#                      + np.random.rand(dim) * (pX[i] - Xnew22))
#             x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
#             fit[i] = fobj(x[i])

#         # ── Thief with Lévy flight + Nonlinear weight ──
#         for i in range(min(19, N), N):
#             beta_levy = 2 * np.random.rand()
#             sigma_u = ((math_gamma(1 + beta_levy) * np.sin(np.pi * beta_levy / 2))
#                        / (math_gamma((1 + beta_levy) / 2) * beta_levy
#                           * 2 ** (0.5 * (beta_levy - 1)))) ** (1 / beta_levy)
#             u = np.random.randn() * sigma_u
#             v = np.random.rand()
#             r5 = np.random.rand()
#             levy = 0.01 * u * r5 / (np.abs(v) ** (1 / beta_levy) + 1e-30)

#             exp_term = 2 * (1 - t / max_iter)
#             w_nl = (np.exp(exp_term) - np.exp(-exp_term)) / (np.exp(exp_term) + np.exp(-exp_term))

#             x[i] = levy * bestX + np.random.randn(dim) * (
#                 np.abs(pX[i] - bestXX) + np.abs(pX[i] - w_nl * bestX)) / 2
#             x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
#             fit[i] = fobj(x[i])

#         # ── Opposition-based learning for worst individuals ──
#         if t % 5 == 0:
#             n_obl = max(1, N // 10)
#             worst_indices = np.argsort(fit)[-n_obl:]
#             for i in worst_indices:
#                 x_obl = lb_vec + ub_vec - pX[i]
#                 x_obl = _discrete_repair(x_obl, lb_vec, ub_vec)
#                 f_obl = fobj(x_obl)
#                 if f_obl < fit[i]:
#                     x[i] = x_obl
#                     fit[i] = f_obl

#         # Update personal & global best
#         XX = pX.copy()
#         for i in range(N):
#             if fit[i] < pFit[i]:
#                 pFit[i] = fit[i]
#                 pX[i] = x[i].copy()
#             if pFit[i] < fMin:
#                 fMin = pFit[i]
#                 bestX = pX[i].copy()

#         # ── Greedy neighbourhood search on global best ──
#         if t % greedy_interval == 0 or t == max_iter:
#             bestX_new, fMin_new = _greedy_neighbourhood(
#                 bestX, fMin, dim, lb_val, ub_val, fobj)
#             if fMin_new < fMin:
#                 fMin = fMin_new
#                 bestX = bestX_new

#         convergence[t - 1] = fMin

#     return fMin, bestX, convergence



# """
# IDBO – Improved Dung Beetle Optimizer for Discrete Task Assignment
#   1. Bernoulli chaotic map initialization
#   2. Variable spiral strategy (brood-ball phase)
#   3. Lévy flight (thief phase)
#   4. Nonlinear weight (thief phase)
#   5. Discrete rounding repair after each position update
#   6. Adaptive Random Neighbourhood Search on global best (Smooth convergence)
#   7. Opposition-based learning for population diversity

# Adapted from BCLDBO for discrete UAV task assignment.
# """
# import numpy as np
# from math import gamma as math_gamma
# from utils import initialization, bernoulli_map, bounds


# def _discrete_repair(x, lb_vec, ub_vec):
#     """Round to nearest integer and clip to bounds – key for discrete problems."""
#     return np.clip(np.round(x), lb_vec, ub_vec)


# def _adaptive_random_neighbourhood(bestX, fMin, dim, lb_val, ub_val, fobj, t, max_iter):
#     """
#     Adaptive Random Neighbourhood Search (ARNS).
#     Replaces the periodic full-greedy search to avoid step-wise (staircase) convergence.
#     Tests random single-dimension mutations every iteration. The number of samples 
#     increases adaptively to ensure smooth convergence and strong final exploitation.
#     """
#     improved = bestX.copy()
#     improved_fit = fMin
    
#     # 随迭代次数自适应增加局部搜索强度（初期重探索，后期重开发）
#     min_samples = 2
#     max_samples = 10
#     num_samples = int(min_samples + (max_samples - min_samples) * (t / max_iter))
    
#     for _ in range(num_samples):
#         # 随机挑选 1 个维度
#         d = np.random.randint(dim)
#         # 随机挑选 1 个可能的值（无人机编号）
#         v = np.random.randint(int(lb_val), int(ub_val) + 1)
        
#         if v == improved[d]:
#             continue
            
#         candidate = improved.copy()
#         candidate[d] = float(v)
#         f_cand = fobj(candidate)
        
#         if f_cand < improved_fit:
#             improved_fit = f_cand
#             improved = candidate.copy()
            
#     return improved, improved_fit


# def IDBO(N, max_iter, lb, ub, dim, fobj):
#     P_percent = 0.2
#     pNum = round(N * P_percent)

#     lb_val = float(lb)
#     ub_val = float(ub)
#     lb_vec = np.full(dim, lb_val)
#     ub_vec = np.full(dim, ub_val)

#     # ── Bernoulli chaotic map + standard initialization ──
#     x = initialization(N, dim, ub, lb)
#     x = _discrete_repair(x, lb_vec, ub_vec)
#     XSi = bernoulli_map(N, dim, ub, lb)
#     XSi = _discrete_repair(XSi, lb_vec, ub_vec)

#     fit = np.array([fobj(x[i]) for i in range(N)])

#     pFit = fit.copy()
#     pX = x.copy()
#     XX = pX.copy()

#     bestI = np.argmin(fit)
#     fMin = fit[bestI]
#     bestX = x[bestI].copy()

#     convergence = np.zeros(max_iter)

#     for t in range(1, max_iter + 1):
#         B = np.argmax(fit)
#         worse = x[B].copy()
#         r2 = np.random.rand()

#         # ── Ball-rolling (producers) ──
#         for i in range(pNum):
#             if r2 < 0.9:
#                 a_val = 1 if np.random.rand() > 0.1 else -1
#                 b_val = np.random.rand()
#                 x[i] = pX[i] + b_val * np.abs(pX[i] - worse) + a_val * 0.1 * XX[i]
#             else:
#                 aaa = np.random.randint(1, 181)
#                 if aaa in (0, 90, 180):
#                     x[i] = pX[i].copy()
#                 theta = aaa * np.pi / 180
#                 x[i] = pX[i] + np.tan(theta) * np.abs(pX[i] - XX[i])
#             x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
#             fit[i] = fobj(x[i])

#         bestII = np.argmin(fit)
#         bestXX = x[bestII].copy()

#         R = 1 - t / max_iter
#         Xnew1 = _discrete_repair(bestXX * (1 - R), lb_vec, ub_vec)
#         Xnew2 = _discrete_repair(bestXX * (1 + R), lb_vec, ub_vec)
#         Xnew11 = _discrete_repair(bestX * (1 - R), lb_vec, ub_vec)
#         Xnew22 = _discrete_repair(bestX * (1 + R), lb_vec, ub_vec)

#         # ── Brood ball with Variable Spiral Strategy ──
#         k_spiral = 5
#         l_spiral = 2 * np.random.rand() - 1
#         z_spiral = np.exp(k_spiral * np.cos(np.pi * (t / max_iter)))
#         spiral_factor = np.exp(z_spiral * l_spiral) * np.cos(2 * np.pi * l_spiral)

#         for i in range(pNum, min(12, N)):
#             x[i] = bestXX + (spiral_factor * np.random.rand(dim) * (pX[i] - Xnew1)
#                               + spiral_factor * np.random.rand(dim) * (pX[i] - Xnew2))
#             x[i] = _discrete_repair(x[i], Xnew1, Xnew2)
#             fit[i] = fobj(x[i])

#         # ── Small DBO with Spiral Strategy ──
#         k_spiral = 5
#         l_spiral = 2 * np.random.rand() - 1
#         z_spiral = np.exp(k_spiral * np.cos(np.pi * (t / max_iter)))
#         spiral_factor = np.exp(z_spiral * l_spiral) * np.cos(2 * np.pi * l_spiral)

#         for i in range(min(12, N), min(19, N)):
#             x[i] = (spiral_factor * pX[i]
#                      + np.random.randn() * (pX[i] - Xnew11)
#                      + np.random.rand(dim) * (pX[i] - Xnew22))
#             x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
#             fit[i] = fobj(x[i])

#         # ── Thief with Lévy flight + Nonlinear weight ──
#         for i in range(min(19, N), N):
#             beta_levy = 2 * np.random.rand()
#             sigma_u = ((math_gamma(1 + beta_levy) * np.sin(np.pi * beta_levy / 2))
#                        / (math_gamma((1 + beta_levy) / 2) * beta_levy
#                           * 2 ** (0.5 * (beta_levy - 1)))) ** (1 / beta_levy)
#             u = np.random.randn() * sigma_u
#             v = np.random.rand()
#             r5 = np.random.rand()
#             levy = 0.01 * u * r5 / (np.abs(v) ** (1 / beta_levy) + 1e-30)

#             exp_term = 2 * (1 - t / max_iter)
#             w_nl = (np.exp(exp_term) - np.exp(-exp_term)) / (np.exp(exp_term) + np.exp(-exp_term))

#             x[i] = levy * bestX + np.random.randn(dim) * (
#                 np.abs(pX[i] - bestXX) + np.abs(pX[i] - w_nl * bestX)) / 2
#             x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
#             fit[i] = fobj(x[i])

#         # ── Opposition-based learning for worst individuals ──
#         # 修改为概率触发，彻底消除任何周期性的阶跃影响
#         if np.random.rand() < 0.2:
#             n_obl = max(1, N // 10)
#             worst_indices = np.argsort(fit)[-n_obl:]
#             for i in worst_indices:
#                 x_obl = lb_vec + ub_vec - pX[i]
#                 x_obl = _discrete_repair(x_obl, lb_vec, ub_vec)
#                 f_obl = fobj(x_obl)
#                 if f_obl < fit[i]:
#                     x[i] = x_obl
#                     fit[i] = f_obl

#         # Update personal & global best
#         XX = pX.copy()
#         for i in range(N):
#             if fit[i] < pFit[i]:
#                 pFit[i] = fit[i]
#                 pX[i] = x[i].copy()
#             if pFit[i] < fMin:
#                 fMin = pFit[i]
#                 bestX = pX[i].copy()

#         # ── 自适应随机邻域搜索 (Adaptive Random Neighbourhood Search) ──
#         # 取代原有的全量贪婪搜索，让曲线自然过渡、平滑收敛，并且仍然具有极高的寻优能力
#         bestX_new, fMin_new = _adaptive_random_neighbourhood(
#             bestX, fMin, dim, lb_val, ub_val, fobj, t, max_iter)
#         if fMin_new < fMin:
#             fMin = fMin_new
#             bestX = bestX_new

#         convergence[t - 1] = fMin

#     return fMin, bestX, convergence



"""
IDBO – Improved Dung Beetle Optimizer for Discrete Task Assignment
  1. Bernoulli chaotic map initialization
  2. Variable spiral strategy (brood-ball phase)
  3. Lévy flight (thief phase)
  4. Nonlinear weight (thief phase)
  5. Discrete rounding repair after each position update
  6. Adaptive Random Neighbourhood Search on global best (Smooth convergence)
  7. Opposition-based learning for population diversity

Adapted from BCLDBO for discrete UAV task assignment.
"""
import numpy as np
from math import gamma as math_gamma
from utils import initialization, bernoulli_map, bounds


def _discrete_repair(x, lb_vec, ub_vec):
    """Round to nearest integer and clip to bounds – key for discrete problems."""
    return np.clip(np.round(x), lb_vec, ub_vec)


def _adaptive_random_neighbourhood(bestX, fMin, dim, lb_val, ub_val, fobj, t, max_iter):
    """
    Adaptive Random Neighbourhood Search (ARNS).
    Replaces the periodic full-greedy search to avoid step-wise (staircase) convergence.
    Tests random single-dimension mutations every iteration. The number of samples 
    increases adaptively to ensure smooth convergence and strong final exploitation.
    """
    improved = bestX.copy()
    improved_fit = fMin
    
    # 【修改】：提高基础采样数，保证早期也有足够的开发力度加速收敛
    # 随迭代次数自适应增加局部搜索强度（初期重探索，后期重开发）
    min_samples = 6
    max_samples = 15
    num_samples = int(min_samples + (max_samples - min_samples) * (t / max_iter))
    
    for _ in range(num_samples):
        # 随机挑选 1 个维度
        d = np.random.randint(dim)
        # 随机挑选 1 个可能的值（无人机编号）
        v = np.random.randint(int(lb_val), int(ub_val) + 1)
        
        if v == improved[d]:
            continue
            
        candidate = improved.copy()
        candidate[d] = float(v)
        f_cand = fobj(candidate)
        
        if f_cand < improved_fit:
            improved_fit = f_cand
            improved = candidate.copy()
            
    return improved, improved_fit


def IDBO(N, max_iter, lb, ub, dim, fobj):
    P_percent = 0.2
    pNum = round(N * P_percent)

    lb_val = float(lb)
    ub_val = float(ub)
    lb_vec = np.full(dim, lb_val)
    ub_vec = np.full(dim, ub_val)

    # ── Bernoulli chaotic map + standard initialization ──
    x = initialization(N, dim, ub, lb)
    x = _discrete_repair(x, lb_vec, ub_vec)
    XSi = bernoulli_map(N, dim, ub, lb)
    XSi = _discrete_repair(XSi, lb_vec, ub_vec)

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
                a_val = 1 if np.random.rand() > 0.1 else -1
                b_val = np.random.rand()
                x[i] = pX[i] + b_val * np.abs(pX[i] - worse) + a_val * 0.1 * XX[i]
            else:
                aaa = np.random.randint(1, 181)
                if aaa in (0, 90, 180):
                    x[i] = pX[i].copy()
                theta = aaa * np.pi / 180
                x[i] = pX[i] + np.tan(theta) * np.abs(pX[i] - XX[i])
            x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
            fit[i] = fobj(x[i])

        bestII = np.argmin(fit)
        bestXX = x[bestII].copy()

        R = 1 - t / max_iter
        Xnew1 = _discrete_repair(bestXX * (1 - R), lb_vec, ub_vec)
        Xnew2 = _discrete_repair(bestXX * (1 + R), lb_vec, ub_vec)
        Xnew11 = _discrete_repair(bestX * (1 - R), lb_vec, ub_vec)
        Xnew22 = _discrete_repair(bestX * (1 + R), lb_vec, ub_vec)

        # ── Brood ball with Variable Spiral Strategy ──
        k_spiral = 5
        l_spiral = 2 * np.random.rand() - 1
        z_spiral = np.exp(k_spiral * np.cos(np.pi * (t / max_iter)))
        spiral_factor = np.exp(z_spiral * l_spiral) * np.cos(2 * np.pi * l_spiral)

        for i in range(pNum, min(12, N)):
            x[i] = bestXX + (spiral_factor * np.random.rand(dim) * (pX[i] - Xnew1)
                              + spiral_factor * np.random.rand(dim) * (pX[i] - Xnew2))
            x[i] = _discrete_repair(x[i], Xnew1, Xnew2)
            fit[i] = fobj(x[i])

        # ── Small DBO with Spiral Strategy ──
        k_spiral = 5
        l_spiral = 2 * np.random.rand() - 1
        z_spiral = np.exp(k_spiral * np.cos(np.pi * (t / max_iter)))
        spiral_factor = np.exp(z_spiral * l_spiral) * np.cos(2 * np.pi * l_spiral)

        for i in range(min(12, N), min(19, N)):
            x[i] = (spiral_factor * pX[i]
                     + np.random.randn() * (pX[i] - Xnew11)
                     + np.random.rand(dim) * (pX[i] - Xnew22))
            x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
            fit[i] = fobj(x[i])

        # ── Thief with Lévy flight + Nonlinear weight ──
        for i in range(min(19, N), N):
            # 【修改】：固定莱维常数，生成与dim同维度的莱维步长，并修正更新公式
            beta_levy = 1.5 
            sigma_u = ((math_gamma(1 + beta_levy) * np.sin(np.pi * beta_levy / 2))
                       / (math_gamma((1 + beta_levy) / 2) * beta_levy
                          * 2 ** (0.5 * (beta_levy - 1)))) ** (1 / beta_levy)
            
            u = np.random.randn(dim) * sigma_u
            v = np.random.randn(dim)
            levy = 0.01 * u / (np.abs(v) ** (1 / beta_levy) + 1e-30)

            exp_term = 2 * (1 - t / max_iter)
            w_nl = (np.exp(exp_term) - np.exp(-exp_term)) / (np.exp(exp_term) + np.exp(-exp_term))

            # 【关键】：保留 bestX 作为基准点，用 levy 作为差值向量的变异步长
            x[i] = bestX + levy * (np.abs(pX[i] - bestXX) + np.abs(pX[i] - w_nl * bestX)) / 2
            x[i] = _discrete_repair(x[i], lb_vec, ub_vec)
            fit[i] = fobj(x[i])

        # ── Opposition-based learning for worst individuals ──
        # 修改为概率触发，彻底消除任何周期性的阶跃影响
        if np.random.rand() < 0.2:
            n_obl = max(1, N // 10)
            worst_indices = np.argsort(fit)[-n_obl:]
            for i in worst_indices:
                x_obl = lb_vec + ub_vec - pX[i]
                x_obl = _discrete_repair(x_obl, lb_vec, ub_vec)
                f_obl = fobj(x_obl)
                if f_obl < fit[i]:
                    x[i] = x_obl
                    fit[i] = f_obl

        # Update personal & global best
        XX = pX.copy()
        for i in range(N):
            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i] = x[i].copy()
            if pFit[i] < fMin:
                fMin = pFit[i]
                bestX = pX[i].copy()

        # ── 自适应随机邻域搜索 (Adaptive Random Neighbourhood Search) ──
        # 取代原有的全量贪婪搜索，让曲线自然过渡、平滑收敛，并且仍然具有极高的寻优能力
        bestX_new, fMin_new = _adaptive_random_neighbourhood(
            bestX, fMin, dim, lb_val, ub_val, fobj, t, max_iter)
        if fMin_new < fMin:
            fMin = fMin_new
            bestX = bestX_new

        convergence[t - 1] = fMin

    return fMin, bestX, convergence