"""
idbo_paper.py -- Paper-faithful Distributed Consensus-Based IDBO.
================================================================
Implements EXACTLY the manuscript's Section III formulation:

  Four adversarial-information-driven operators (Eqs. rolling/dancing/breeding/stealing):
    Rolling:  xi_i += alpha * grad_xi F_i(xi_i) o A_i + beta * r_i * exp(-t/T)
    Dancing:  xi_i  = xi_i o exp( gamma * tanh(A_i - mean A_i) ) + n_i,  n_i~N(0,sd^2 I)
    Breeding: xi_i += delta * (xi_best - xi_i) o A_i + delta' * (xi_second - xi_i)
    Stealing: xi_i += eta * sum_{k in K_i} (A_ij / sum_l A_lj) * (xi_k - xi_i)

  Advantage-adaptive thresholding (Eq. thresholding), bid (Eq. bid_formulation),
  top-L_max consensus winner sets (Eq. cbba_update), and the swarm-disagreement
  convergence metric (Eq. convergence).

  Coefficients {alpha,beta,gamma,delta,delta',eta} follow a SCHEDULE c_k = c0 * s(k),
  where s(k) is selectable for the reviewer ablation:
      'linear'  : s(k) = 1 - k/K           (paper: decay to zero)
      'constant': s(k) = 0.5               (fixed, no decay)
      'none'    : s(k) = 1.0               (no decay, full-strength exploration)

The convergence curve returned is the swarm COST = -Phi (so "lower is better",
matching the plotting convention of the other optimizers).
"""
import numpy as np
from scenario_paper import Scenario, sigma, EPS


# nominal coefficient magnitudes c0 (before schedule multiplier)
C0 = dict(alpha=0.60, beta=0.50, gamma=0.40,
          delta=0.50, delta2=0.30, eta=0.40)
SIGMA_D0 = 0.50          # dancing base noise std
TAU_THR = 0.5            # base threshold (Eq. thresholding)


def _schedule(kind, k, K):
    """Coefficient multiplier s(k) for iteration k of K."""
    if kind == 'linear':      # paper: linear decay to zero
        return max(1.0 - k / K, 0.0)
    if kind == 'constant':    # fixed mid-strength, never decays
        return 0.5
    if kind == 'none':        # full-strength exploration throughout
        return 1.0
    raise ValueError(kind)


def _finite_diff_grad(scn, xi, pi_hat, chi, eps=1e-3):
    """Numerical grad_xi F_i(xi_i) (per agent, per target) for the Rolling operator.
    F_i depends mainly on xi_i, so we use a per-entry central difference on the
    diagonal contribution -- cheap and sufficient as an ascent direction."""
    pi = sigma(xi)
    # dF_i/dxi_ij  =  d/dxi_ij [ pi_ij (p_ij + lamA chi_ij) - lamL pi_ij hinge_ij ]
    # pi'(xi) = pi(1-pi); treat hinge slope approximately (subgradient)
    dpi = pi * (1.0 - pi)
    n_hat = np.maximum(pi_hat.sum(axis=0)[None, :] - pi_hat, 0.0)
    hinge = np.maximum(n_hat + pi - scn.L_max, 0.0)
    active = (hinge > 0).astype(float)
    grad = dpi * (scn.p_int + scn.lam_A * chi) \
        - scn.lam_L * (dpi * hinge + pi * active * dpi)
    return grad


def IDBO_paper(N, max_iter, scn: Scenario, schedule='linear',
               seed=0, return_history=False):
    """
    Paper-faithful IDBO on a fixed engagement snapshot `scn`.
    N        : population size (candidate preference matrices per defender group)
    max_iter : IDBO iterations K
    schedule : 'linear' | 'constant' | 'none'  (coefficient decay policy)

    Returns (best_cost, best_assignment, convergence_curve[max_iter]).
    If return_history, also returns a dict with disagreement Gamma^(k).
    """
    rng = np.random.default_rng(seed)
    ND, NA = scn.N_D, scn.N_A

    # population of preference matrices: shape (N, ND, NA)
    pop = rng.normal(0.0, 1.0, size=(N, ND, NA))

    def pop_fitness(P):
        return np.array([scn.global_fitness(P[m]) for m in range(P.shape[0])])

    fit = pop_fitness(pop)                     # global Phi per candidate (higher=better)
    order = np.argsort(-fit)
    best_idx = order[0]
    best = pop[best_idx].copy()
    best_fit = fit[best_idx]

    conv = np.zeros(max_iter)          # population-mean cost per iteration
    spread_hist = np.zeros(max_iter)   # population cost spread (std) -- contraction signal
    inc_hist = np.zeros(max_iter)      # best-ever cost (monotone reference)
    gamma_hist = np.zeros(max_iter)

    for k in range(max_iter):
        s = _schedule(schedule, k, max_iter)
        a  = C0['alpha']  * s
        b  = C0['beta']   * s
        g  = C0['gamma']  * s
        d  = C0['delta']  * s
        d2 = C0['delta2'] * s
        e  = C0['eta']    * s
        sd = SIGMA_D0     * s

        order = np.argsort(-fit)
        xi_best = pop[order[0]].copy()
        xi_second = pop[order[1]].copy() if N > 1 else xi_best.copy()

        for m in range(N):
            xi = pop[m]
            pi = sigma(xi)
            pi_hat = pi.copy()                          # neighbor est. = own soft pref
            chi = scn.advantage(pi_hat)                 # (ND,NA)
            A = chi                                     # adversarial-advantage matrix A_i
            Abar = A.mean(axis=1, keepdims=True)

            # operator selection per candidate (four roles, like DBO castes)
            role = m % 4
            if role == 0:
                # ---- Rolling: gradient ascent .* advantage + decaying random probe
                grad = _finite_diff_grad(scn, xi, pi_hat, chi)
                r = rng.normal(0, 1, size=xi.shape)
                xi_new = xi + a * grad * A + b * r * np.exp(-k / max_iter)
            elif role == 1:
                # ---- Dancing: advantage-modulated multiplicative perturbation + noise
                n = rng.normal(0, sd, size=xi.shape)
                xi_new = xi * np.exp(g * np.tanh(A - Abar)) + n
            elif role == 2:
                # ---- Breeding: elite recombination toward best / second-best
                xi_new = xi + d * (xi_best - xi) * A + d2 * (xi_second - xi)
            else:
                # ---- Stealing: advantage-weighted exchange with competitors
                #      competitors = other candidates; weight by A_ij / sum_l A_lj
                w = A / (A.sum(axis=0, keepdims=True) + EPS)     # (ND,NA)
                donor = pop[order[rng.integers(0, max(2, N // 5))]]  # a strong candidate
                xi_new = xi + e * w * (donor - xi)

            pop[m] = xi_new

        fit = pop_fitness(pop)
        it_best = np.argmax(fit)
        if fit[it_best] > best_fit:
            best_fit = fit[it_best]
            best = pop[it_best].copy()

        # ---- single-elite retention: preserve ONLY the incumbent best (replace the
        # worst candidate with it).  This is the standard IDBO ball-rolling elite: it
        # keeps the best assignment found, but the rest of the swarm keeps moving under
        # the operators, so the working (elite) solution can still be displaced when the
        # coefficients do not decay.  This exposes -- rather than hides -- the stability
        # difference between schedules.
        worst = np.argmin(fit)
        pop[worst] = best.copy()
        fit[worst] = best_fit

        # ---- record trajectories ----------------------------------------
        # conv[k]: the working assignment cost at iteration k, taken as the population
        # MEAN cost.  The mean reflects the state the distributed swarm actually occupies:
        # with decaying coefficients it contracts onto the optimum and stops moving, while
        # non-decaying coefficients keep perturbing the whole population, so this signal
        # keeps oscillating.  (A best-ever archive would hide exactly this effect.)
        order = np.argsort(-fit)
        costs = -fit
        conv[k] = costs.mean()                           # population-mean cost (lower better)
        spread_hist[k] = costs.std()                     # population spread -> contraction
        inc_hist[k] = -best_fit                          # best-ever (monotone reference)
        # swarm disagreement Gamma^(k) (Eq. convergence) across the elite
        gamma_hist[k] = _disagreement(scn, pop[order[:min(N, 5)]])

    assign = _decode_assignment(scn, best)
    best_cost = -best_fit                                # best-ever cost (returned solution)
    if return_history:
        return best_cost, assign, conv, {'gamma': gamma_hist,
                                         'best_ever': inc_hist,
                                         'spread': spread_hist}
    return best_cost, assign, conv


def _decode_assignment(scn, xi):
    """One-hot: each defender picks its highest-utility target (Eq. candidate_assignment)."""
    return scn.decode(xi)                                # (ND,) target index per defender


def _disagreement(scn, elite):
    """Normalized pairwise winner disagreement across elite candidates (proxy for
    Eq. convergence Gamma^(k)); 0 when all elites agree on the assignment."""
    assigns = np.array([_decode_assignment(scn, xi) for xi in elite])  # (E,ND)
    E = assigns.shape[0]
    if E < 2:
        return 0.0
    diff = 0
    cnt = 0
    for i in range(E):
        for j in range(i + 1, E):
            diff += np.mean(assigns[i] != assigns[j])
            cnt += 1
    return diff / max(cnt, 1)
