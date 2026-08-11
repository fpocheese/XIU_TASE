"""
consensus_idbo.py -- Distributed consensus-based IDBO with an explicit communication
graph and communication-delay model.
=====================================================================================
Reviewer response (Re3_2): rigorous complexity / scalability analysis of the IDBO under
cooperative-deflection environments where COMMUNICATION DELAYS influence distributed
consensus efficiency.

This module implements the manuscript's distributed consensus EXACTLY:

  * Each defender d_i runs a local IDBO preference search and picks ONE candidate target
    (Eq. candidate_assignment):  Xtil_ij = 1[ j = argmax_r u_ir ].
  * It forms a bid on that target (Eq. bid_formulation):
        b_ij = Xtil_ij * u_ij * exp( (chi_ij - mean chi_i) / (std chi_i + eps) ).
  * Each defender keeps a local winner set W_{i,j} per target and, after receiving
    neighbor copies, retains the top-L_max bids (Eq. cbba_update):
        W_{i,j} <- Top_Lmax( W_{i,j} U (U_{l in N_i} W_{l,j}) U {(i,b_ij)} ).
  * A defender displaced from a full winner set drops that target and re-picks next round.
  * Consensus is measured by the swarm disagreement (Eq. convergence):
        Gamma^(k) = (1/|E|) sum_{(i,l) in E} 1[ W_i != W_l ]   ->  <= eps_con.

COMMUNICATION DELAY MODEL.
  On a graph with per-edge integer delay d (rounds), when d_i merges neighbor d_l's winner
  sets at round k it uses the STALE copy W_{l}^{(k-d)} (or the most recent available).
  Packet loss p_loss optionally drops a neighbor message for that round.  This is the
  standard delayed-information consensus setting and directly exposes how delay slows the
  agreement process without changing the fixed point.
"""
import numpy as np
from collections import deque
from scenario_paper import Scenario, sigma, EPS


# ---------------------------------------------------------------- comm graphs
def make_graph(n, topo='rgg', seed=0, radius=0.35, ring_k=2):
    """Return (adjacency list neighbors[i], edge list E, diameter D) for n defenders.
    topo: 'ring' (k-nearest ring), 'rgg' (random geometric graph), 'complete'."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=bool)
    if topo == 'complete':
        A[:] = True
        np.fill_diagonal(A, False)
    elif topo == 'ring':
        for i in range(n):
            for dk in range(1, ring_k + 1):
                A[i, (i + dk) % n] = True
                A[i, (i - dk) % n] = True
    elif topo == 'rgg':
        # random geometric graph on unit square, connected (retry until connected)
        for _ in range(200):
            pts = rng.random((n, 2))
            D2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
            A = (D2 < radius ** 2)
            np.fill_diagonal(A, False)
            if _is_connected(A):
                break
        else:
            # fall back to ring to guarantee connectivity
            A[:] = False
            for i in range(n):
                A[i, (i + 1) % n] = True
                A[i, (i - 1) % n] = True
    else:
        raise ValueError(topo)
    neighbors = [np.where(A[i])[0].tolist() for i in range(n)]
    E = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    D = _diameter(A)
    return neighbors, E, D


def _is_connected(A):
    n = A.shape[0]
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in np.where(A[u])[0]:
            if v not in seen:
                seen.add(int(v))
                stack.append(int(v))
    return len(seen) == n


def _diameter(A):
    """BFS all-pairs longest shortest-path (unweighted). Returns inf-safe int."""
    n = A.shape[0]
    diam = 0
    for s in range(n):
        dist = -np.ones(n, dtype=int)
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in np.where(A[u])[0]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(int(v))
        if (dist < 0).any():
            return n  # disconnected fallback
        diam = max(diam, dist.max())
    return int(diam)


# ---------------------------------------------------------------- winner-set consensus
class DistributedIDBO:
    """Distributed consensus-based IDBO on a fixed engagement snapshot.

    Runs one CBBA-style winner-set consensus with an explicit communication graph and an
    integer per-message delay.  Records per-round disagreement Gamma^(k), communication
    volume, and wall-clock, so complexity / scalability / delay can be quantified.
    """

    def __init__(self, scn: Scenario, neighbors, E, delay=0, p_loss=0.0,
                 local_iters=6, seed=0):
        self.scn = scn
        self.nb = neighbors
        self.E = E
        self.delay = int(delay)
        self.p_loss = float(p_loss)
        self.local_iters = local_iters
        self.rng = np.random.default_rng(seed)
        self.ND, self.NA = scn.N_D, scn.N_A
        self.Lmax = scn.L_max

        # per-defender latent preference (own row), initialized random
        self.xi = self.rng.normal(0, 1, size=(self.ND, self.NA))
        # per-defender local winner set: dict target -> list of (agent, bid), len<=Lmax
        self.W = [{j: [] for j in range(self.NA)} for _ in range(self.ND)]
        # delayed outbox history: for each defender, a deque of past winner-set snapshots
        self.history = [deque(maxlen=self.delay + 1) for _ in range(self.ND)]
        # per-defender advantage row (static snapshot advantage)
        self.chi = scn.chi_static                     # (ND,NA)
        self.p_int = scn.p_int
        self.comm_msgs = 0                            # cumulative neighbor messages merged

    # ---- one defender's local pick + bid (Eqs. candidate_assignment, bid_formulation)
    def _pick_and_bid(self, i, k):
        # local IDBO-style refinement of xi[i] using advantage-weighted ascent + decaying probe
        xi = self.xi[i].copy()
        chi_i = self.chi[i]
        # neighbor-estimated load n_hat_j from current local winner sets
        n_hat = np.array([len(self.W[i][j]) for j in range(self.NA)], dtype=float)
        s = max(1.0 - k / max(self.local_iters * 4, 1), 0.05)
        for _ in range(self.local_iters):
            pi = sigma(xi)
            hinge = np.maximum(n_hat + pi - self.Lmax, 0.0)
            dpi = pi * (1 - pi)
            grad = dpi * (self.p_int[i] + self.scn.lam_A * chi_i) - self.scn.lam_L * dpi * hinge
            xi = xi + 0.5 * s * grad + 0.3 * s * self.rng.normal(0, 1, self.NA) * np.exp(-k / 20.0)
        self.xi[i] = xi
        pi = sigma(xi)
        hinge = np.maximum(n_hat + pi - self.Lmax, 0.0)
        u = pi * (self.p_int[i] + self.scn.lam_A * chi_i) - self.scn.lam_L * pi * hinge
        # forbid targets from which this defender was displaced in a full winner set
        for j in range(self.NA):
            wj = self.W[i][j]
            if len(wj) >= self.Lmax and i not in [a for a, _ in wj]:
                u[j] = -np.inf
        j_star = int(np.argmax(u))
        # bid (Eq. bid_formulation)
        cbar = chi_i.mean()
        cstd = chi_i.std() + EPS
        b = u[j_star] * np.exp((chi_i[j_star] - cbar) / cstd)
        return j_star, max(b, 1e-6)

    # ---- Top-L_max merge of winner sets (Eq. cbba_update) --------------------
    @staticmethod
    def _topL(entries, Lmax):
        """Keep the L_max highest bids; one entry per agent (latest bid wins)."""
        best = {}
        for agent, bid in entries:
            if agent not in best or bid > best[agent]:
                best[agent] = bid
        merged = sorted(best.items(), key=lambda ab: -ab[1])[:Lmax]
        return merged  # list of (agent, bid)

    def _winner_signature(self, i):
        """Hashable signature of defender i's winner sets (agent-id sets per target),
        used to test cross-link agreement for Gamma^(k)."""
        return tuple(frozenset(a for a, _ in self.W[i][j]) for j in range(self.NA))

    def _gamma(self):
        """Swarm disagreement (Eq. convergence): fraction of communication links whose
        winner information differs.  0 == all neighbors agree."""
        sigs = [self._winner_signature(i) for i in range(self.ND)]
        if not self.E:
            return 0.0
        dis = sum(1 for (i, j) in self.E if sigs[i] != sigs[j])
        return dis / len(self.E)

    # ---- one consensus round -------------------------------------------------
    def _round(self, k):
        # (1) every defender picks a target and bids
        for i in range(self.ND):
            j_star, b = self._pick_and_bid(i, k)
            # insert own bid into its own winner set for that target
            self.W[i][j_star] = self._topL(self.W[i][j_star] + [(i, b)], self.Lmax)

        # (2) snapshot current winner sets into each defender's delay history
        for i in range(self.ND):
            self.history[i].append([dict_j.copy() for dict_j in [self.W[i]]][0])

        # (3) delayed neighbor merge: defender i merges the copy of neighbor l that is
        #     `delay` rounds old (or the oldest available early on); packets may be lost.
        newW = [None] * self.ND
        for i in range(self.ND):
            merged = {j: list(self.W[i][j]) for j in range(self.NA)}
            for l in self.nb[i]:
                if self.p_loss > 0 and self.rng.random() < self.p_loss:
                    continue                                   # packet dropped this round
                hist = self.history[l]
                # stale copy: index 0 is oldest kept (== k-delay once filled)
                stale = hist[0] if len(hist) > 0 else self.W[l]
                self.comm_msgs += 1
                for j in range(self.NA):
                    merged[j] = self._topL(merged[j] + list(stale[j]), self.Lmax)
            newW[i] = merged
        self.W = newW

        # (4) displacement: if defender i is not in a target's full winner set, it is
        #     dropped from that target (handled next round via the u[j]=-inf rule).
        return self._gamma()

    # ---- full run ------------------------------------------------------------
    def run(self, max_rounds=200, eps_con=0.0, patience=5):
        """Run consensus until Gamma stays <= eps_con for `patience` consecutive rounds,
        or max_rounds is hit.  Returns dict with per-round gamma, comm volume, rounds."""
        import time
        gamma_hist = []
        t0 = time.perf_counter()
        stable = 0
        rounds_to_consensus = max_rounds
        for k in range(max_rounds):
            g = self._round(k)
            gamma_hist.append(g)
            if g <= eps_con:
                stable += 1
                if stable >= patience:
                    rounds_to_consensus = k + 1
                    break
            else:
                stable = 0
        wall = time.perf_counter() - t0
        assign = self._final_assignment()
        return {
            'gamma': np.array(gamma_hist),
            'rounds': rounds_to_consensus,
            'wall': wall,
            'comm_msgs': self.comm_msgs,
            'assignment': assign,
            'cost': self.scn.assignment_cost(assign),
            'converged': rounds_to_consensus < max_rounds,
        }

    def _final_assignment(self):
        """Decode: each defender -> the target whose winner set it belongs to (or its best
        local pick if it holds no slot)."""
        assign = np.zeros(self.ND, dtype=int)
        for i in range(self.ND):
            held = [j for j in range(self.NA) if i in [a for a, _ in self.W[i][j]]]
            if held:
                assign[i] = held[0]
            else:
                pi = sigma(self.xi[i])
                assign[i] = int(np.argmax(pi * (self.p_int[i] + self.scn.lam_A * self.chi[i])))
        return assign
