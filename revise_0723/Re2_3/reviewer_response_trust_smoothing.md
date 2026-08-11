# Reviewer Response — Text vs. Eq. (42): Convergent Smoothing of Trust

**Reviewer comment (Re2_3).** *Section IV.B contains a logical paradox: the text
describes a monotonic incremental process, while Equation (42) implements a
convergent smoothing process. This divergence between the mathematical definition
and physical intuition renders the mechanism logically inconsistent in maintaining
high-trust stability.*

---

## Response summary

We agree, and we take **Eq. (42) as authoritative**. The mathematics is correct; the
inconsistency was only in the surrounding prose, which described the update as a
monotonic ratchet ("driving `T_i` upward" / "lowers `T_i`"). We kept Eq. (42)
unchanged and **rewrote the text** so the physical description matches the
mathematics. One illustrative figure (Fig. R1) is provided for the response letter
(not added to the main paper).

## 1. Why Eq. (42) is a convergent smoother, not a monotonic increment

Eq. (42):

```
T_i^{(k+1)} = (1 - alpha_T) * T_i^{(k)} + alpha_T * sigma(tau_T * Rtilde_i^{(k)}),   alpha_T in (0,1)
```

This is first-order exponential smoothing toward the bounded, moving target
`T* = sigma(tau_T * Rtilde_i^{(k)}) in (0,1)`. Subtracting the target:

```
T_i^{(k+1)} - T* = (1 - alpha_T) * ( T_i^{(k)} - T* )
```

so the map is a **contraction of modulus (1 - alpha_T) < 1**. Therefore:

1. `T_i^{(k)}` stays in `(0,1)` for all `k`;
2. it **converges** to the equilibrium set by current relative performance, rather
   than growing without bound;
3. sustained above-average performance holds the target above `0.5`, so trust
   **settles at and maintains** a high value — which is exactly what produces stable
   high trust. A genuinely monotonic process would instead saturate at the ceiling
   and lose all sensitivity.

The paradox the reviewer identified was thus a wording error, now removed.

## 2. Manuscript change (Section IV-B, after Eq. (42))

- **Deleted (red strikethrough):** "When `R_i^{(k)} > mu_R`, the sigmoid output
  exceeds 0.5, driving `T_i` upward and reducing exploration; conversely,
  below-average performance lowers `T_i` and amplifies exploration."
- **Added (blue):** a description of Eq. (42) as a first-order
  exponential-smoothing (contraction) update relaxing `T_i` toward the moving
  equilibrium `sigma(tau_T*Rtilde)`, converging (not accumulating), and thereby
  *maintaining* stable high trust. Full wording is in the master response letter
  (`response_to_journal/response_to_reviewers.tex`) and highlighted in
  `new_highlight/main.pdf` (Section IV-B, p. 7).

## 3. Supporting figure (this letter)

**Fig. R1 — `trust_convergent_smoothing.pdf`** (3 panels). Numerically verified:

- **(a)** Sustained above-average return: Eq. (42) converges to and holds a stable
  high-trust equilibrium `T ~ 0.83` (below the ceiling); the monotonic-increment
  reading ratchets up and pins at `1.0`.
- **(b)** Time-varying return: trust smoothly tracks the moving target both **up**
  (max step +0.060) and **down** (max step -0.098) — impossible for a monotonic
  process.
- **(c)** Contraction: trajectories from any initial `T_i^{(0)}` in
  `{0.02, 0.25, 0.5, 0.75, 0.98}` converge to the same equilibrium `T* = 0.769`;
  the deviation decays geometrically with per-step ratio `0.8800 = (1 - alpha_T)`.

**Suggested caption — Fig. R1.**

> Fig. R1. The trust update of Eq. (42) is a convergent smoother (contraction toward
> a bounded, moving equilibrium), not a monotonic increment. (a) Sustained
> above-average return: Eq. (42) converges to and maintains a stable high-trust value
> (T* = sigma(tau_T*Rtilde), dashed), whereas a monotonic-increment reading ratchets
> to and pins at the ceiling. (b) Time-varying return: trust smoothly tracks the
> moving target both up and down. (c) Contraction: trajectories from any initial
> T_i^{(0)} converge to a unique equilibrium, and the deviation decays geometrically
> along the (1-alpha_T)^k envelope. This is exactly the mechanism by which the update
> maintains stable high trust.

---

*Implementation note (internal): figure script is `Re2_3/code/trust_dynamics.py`
(illustrative parameters alpha_T = 0.12, tau_T = 2.0; simulates Eq. (42) directly).*
