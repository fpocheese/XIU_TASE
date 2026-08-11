"""
scenario_paper.py -- Engagement scenario for the paper-faithful IDBO.
=====================================================================
Reviewer-response experiment (Section III, Distributed Consensus-Based IDBO).

This defines the cooperative target-assignment instance the way the PAPER models
it (not the legacy dbo_code cost function):

  * N_D defending UAVs, N_A attacking targets (paper case: 20 vs 8).
  * Each defender d_i holds an unconstrained preference vector xi_i in R^{N_A};
    soft preference pi_ij = sigma(xi_ij).
  * Per-target interception probability p^int_ij from an engagement-geometry model.
  * Combat advantage chi_ij fuses time, relative-velocity compatibility and aspect
    angle, minus local competition                          (Eq. adversarial_advantage).
  * Local utility / fitness with a soft over-saturation hinge  (Eq. local_utility).

Everything here follows the manuscript; the legacy files in ../../dbo_code are left
untouched.
"""
import numpy as np

# ------------------------------------------------------------------ constants
EPS = 1e-8


def sigma(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class Scenario:
    """A fixed engagement snapshot: defender & target kinematics + engagement model."""

    def __init__(self, n_def=20, n_att=8, L_max=3, seed=0,
                 lam_A=0.6, lam_L=1.2, lam_C=0.5,
                 T_c=60.0, v_c=40.0, R_ref=2000.0, vc_close=40.0):
        self.N_D = n_def
        self.N_A = n_att
        self.L_max = L_max
        self.lam_A = lam_A      # advantage weight  (lambda_A in Eq. local_utility)
        self.lam_L = lam_L      # saturation-penalty weight (lambda_L)
        self.lam_C = lam_C      # competition weight (lambda_C in Eq. adversarial_advantage)
        self.T_c = T_c          # time normalization (s)
        self.v_c = v_c          # velocity normalization (m/s)
        self.R_ref = R_ref      # nominal engagement radius (m) -- paper attacker shell ~2 km
        self.vc_close = vc_close  # closing-speed normalization (m/s)

        rng = np.random.default_rng(seed)

        # ---- Initialization following the manuscript's scenario settings ----
        # Defenders: initial radius 52.63--65.44 m, altitude 0 m, speed 20.32--30.02 m/s,
        #            heading gamma_D in [0, 2pi).
        # Targets:   initial radius 1987.72--2145.95 m, altitude 120 m,
        #            speed 30.23--45.87 m/s, heading toward the defended area (origin).
        # A defended area is centered at the origin; defenders ring it closely while the
        # attacking swarm approaches from an outer shell, giving a realistic 3-D
        # interception geometry with varied ranges and aspect angles.
        rD = rng.uniform(52.63, 65.44, n_def)                  # m
        thD = rng.uniform(0, 2 * np.pi, n_def)
        self.pD = np.column_stack([rD * np.cos(thD),
                                   rD * np.sin(thD),
                                   np.zeros(n_def)])           # altitude 0 m
        rA = rng.uniform(1987.72, 2145.95, n_att)              # m
        thA = rng.uniform(0, 2 * np.pi, n_att)
        self.pT = np.column_stack([rA * np.cos(thA),
                                   rA * np.sin(thA),
                                   np.full(n_att, 120.0)])     # altitude 120 m

        # defender velocities: heading in [0,2pi), small climb, speed 20.32--30.02 m/s
        speedD = rng.uniform(20.32, 30.02, n_def)
        headD = rng.uniform(0, 2 * np.pi, n_def)
        self.vD = np.column_stack([speedD * np.cos(headD),
                                   speedD * np.sin(headD),
                                   np.zeros(n_def)])
        # target velocities: directed toward the defended area (origin), 30.23--45.87 m/s
        speedT = rng.uniform(30.23, 45.87, n_att)
        dir_to_origin = -self.pT / (np.linalg.norm(self.pT, axis=1, keepdims=True) + EPS)
        self.vT = dir_to_origin * speedT[:, None]

        self._precompute()

    # -------------------------------------------------- engagement quantities
    def _precompute(self):
        """Range, closing time, aspect angle, and interception probability p^int_ij.
        These depend only on the (fixed) snapshot, so compute once."""
        ND, NA = self.N_D, self.N_A
        rel = self.pT[None, :, :] - self.pD[:, None, :]          # (ND,NA,3) m
        rng = np.linalg.norm(rel, axis=2)                        # (ND,NA) m
        self.rng_m = rng
        los = rel / (rng[:, :, None] + EPS)

        # aspect angle between defender velocity and LOS to the target: a defender
        # heading toward a target engages it more effectively (primary discriminator).
        vD_norm = self.vD / (np.linalg.norm(self.vD, axis=1, keepdims=True) + EPS)
        cos_asp = np.sum(vD_norm[:, None, :] * los, axis=2)      # (ND,NA)
        self.cos_asp = cos_asp
        self.aspect = np.arccos(np.clip(cos_asp, -1, 1))

        # relative-velocity magnitude (speed compatibility) for the advantage
        vrel = self.vD[:, None, :] - self.vT[None, :, :]         # (ND,NA,3) m/s
        self.dv = np.linalg.norm(vrel, axis=2)                   # (ND,NA) m/s

        # required turn-in effort -> intercept-time proxy: a defender already pointing at
        # the target (cos_asp near 1) reaches it sooner; misaligned defenders need longer.
        speedD = np.linalg.norm(self.vD, axis=1, keepdims=True)  # (ND,1)
        turn_factor = 0.5 * (1.0 - cos_asp) + 0.1                # in [0.1,1.1]
        self.dt = rng * turn_factor / (speedD + EPS)             # intercept-time proxy (s)

        # interception probability: closer range + favorable aspect => higher.
        # Range normalized by the nominal engagement radius R_ref (paper attacker shell ~2 km);
        # aspect is the dominant, well-spread discriminator across defender headings.
        rng_term = np.exp(-rng / (2.0 * self.R_ref))             # gentle range decay
        asp_term = np.clip(0.5 + 0.5 * cos_asp, 0, 1)            # in [0,1], head-on favored
        self.p_int = np.clip(0.10 + 0.88 * (0.25 * rng_term + 0.75 * asp_term ** 1.5),
                             0.02, 0.99)                          # (ND,NA)

        # static advantage matrix (uniform neighbor-load proxy), used by the decoded
        # assignment_cost as a tactical bonus; recomputed dynamically inside advantage().
        self.chi_static = self.advantage(np.full((ND, NA), 0.5))

    # -------------------------------------------------- combat advantage chi_ij
    def advantage(self, pi_hat):
        """Eq. adversarial_advantage.  pi_hat: (ND,NA) neighbor soft-preference estimate
        (here shared across defenders as the swarm mean)."""
        time_term = np.exp(-self.dt / self.T_c)
        vel_term = np.clip(1.0 - self.dv / (2.0 * self.v_c), 0, 1)     # dv in m/s
        geo_term = np.clip(self.cos_asp, 0, 1)
        base = time_term * vel_term * geo_term                        # (ND,NA)

        # local competition term
        p = self.p_int                                                # (ND,NA)
        comp = np.zeros_like(base)
        col_pi = pi_hat.mean(axis=0)                                  # (NA,) neighbor load proxy
        for j in range(self.N_A):
            denom = p[:, j][:, None] + p[:, j][None, :] + EPS         # (ND,ND)
            ratio = p[:, j][None, :] / denom                          # (ND,ND) p_lj / (p_ij+p_lj)
            np.fill_diagonal(ratio, 0.0)
            comp[:, j] = self.lam_C * (ratio * col_pi[j]).sum(axis=1) / max(self.N_D - 1, 1)
        return base - comp

    # -------------------------------------------------- fitness  F_i(xi_i)
    def fitness_and_utility(self, xi, pi_hat, chi):
        """Eq. local_utility.  xi: (ND,NA) preference; returns per-agent fitness (ND,)
        and per-(i,j) utility (ND,NA).  The utility drives per-defender target choice;
        the soft over-saturation hinge only discourages piling onto an over-subscribed
        target.  n_hat is the neighbor-estimated load (excluding self)."""
        pi = sigma(xi)                                               # (ND,NA)
        n_hat = np.maximum(pi_hat.sum(axis=0)[None, :] - pi_hat, 0.0)
        hinge = np.maximum(n_hat + pi - self.L_max, 0.0)            # [.]_+
        u = pi * (self.p_int + self.lam_A * chi) - self.lam_L * pi * hinge
        return u.sum(axis=1), u

    # -------------------------------------------------- decoded-assignment objective
    def assignment_cost(self, assign):
        """Well-posed objective on a DECODED one-hot assignment (Eq. optimization_
        formulation / Eq. candidate_assignment): minimize expected surviving targets
        plus a capacity-overflow penalty.

        assign : (N_D,) target index chosen by each defender.
        Lower is better.  A target is 'covered' by its assigned defenders; its survival
        probability is prod_{i->j}(1 - p_int_ij).  Overloading a target beyond L_max is
        penalized; leaving a target uncovered leaves survival = 1."""
        surv = np.ones(self.N_A)
        load = np.zeros(self.N_A, dtype=int)
        adv_bonus = 0.0
        for i in range(self.N_D):
            j = assign[i]
            surv[j] *= (1.0 - self.p_int[i, j])
            load[j] += 1
            adv_bonus += self.lam_A * self.chi_static[i, j]
        overflow = np.maximum(load - self.L_max, 0).sum()
        # expected survivors (want small) + capacity penalty - advantage reward
        return surv.sum() + self.lam_L * overflow - 0.02 * adv_bonus

    def global_fitness(self, xi):
        """Swarm fitness Phi used as the convergence metric.  We decode xi to a one-hot
        assignment and return Phi = -cost (higher = better) so the optimizer maximizes."""
        assign = self.decode(xi)
        return -self.assignment_cost(assign)

    def decode(self, xi):
        """One-hot target choice per defender = argmax utility (Eq. candidate_assignment)."""
        pi = sigma(xi)
        pi_hat = pi.copy()
        chi = self.advantage(pi_hat)
        _, u = self.fitness_and_utility(xi, pi_hat, chi)
        return np.argmax(u, axis=1)
