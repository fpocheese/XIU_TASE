# Reviewer Response — Pseudocode / Algorithm Boxes for IDBO and ART-MAPPO

**Reviewer comment (Re1_3).** *Include pseudocode or algorithm boxes for IDBO and
ART-MAPPO to improve reproducibility.*

---

## Response summary

We thank the reviewer for this helpful suggestion. We have added two formal
algorithm boxes to the manuscript:

- **Algorithm 1 — Distributed Consensus-Based IDBO** (end of Section III-B), and
- **Algorithm 2 — ART-MAPPO Training (CTDE)** (end of Section IV-D).

Because the comment explicitly asks for the boxes to improve reproducibility, these
are added to the **main paper** (not only the response letter). Every line of both
algorithms is **cross-referenced to the defining equation** already in the paper, so
a reader can reproduce each step from the corresponding formula. The boxes are typeset
with the `algorithm`/`algorithmic` environments already loaded by the IEEEtran
template, and are highlighted in blue in the revised manuscript.

## Algorithm 1 — Distributed Consensus-Based IDBO (per interceptor)

Structure: three interleaved phases per outer iteration, run until the swarm
disagreement drops below the threshold.

- **Inputs:** interceptor/target sets, probability matrix `P` (Eq. 8), neighbors
  `N_i`, capacity `L_max`, IDBO iterations `T`, population size, threshold `eps`.
- **Phase 1 (local optimization):** decaying coefficients `c_t=c0(1-t/T)`; fitness
  (Eq. 13) and advantage (Eq. 15); four operators
  Rolling/Dancing/Breeding/Stealing (Eqs. 16–19); advantage-adaptive binarization
  (Eq. 20).
- **Phase 2 (auction-based consensus):** bids (Eq. 21); exchange winner lists and
  update the top-`L_max` set (Eq. 22).
- **Phase 3 (advantage update):** refresh advantages from the consensus assignment;
  compute disagreement `Gamma` (Eq. 26).
- **Output:** converged assignment `X*`.

## Algorithm 2 — ART-MAPPO Training (CTDE)

Structure: decentralized rollout, trust update from episode returns, then
centralized optimization.

- **Inputs:** actor/critic params, clip `eps`, dual-clip floor `c`, target KL, rates
  `alpha_T, rho, tau_T`, epochs `K`.
- **Rollout (decentralized):** attention-residual features (Eqs. 28–31), temporal
  state via GRU (Eq. 36), Gaussian policy (Eq. 37), trust-blended action
  (Eqs. 43, 44 with `beta(T_i)=1-T_i`), clip to envelope.
- **Trust update:** running stats (Eqs. 39, 40), normalization (Eq. 41), first-order
  exponential smoothing (Eq. 42).
- **Centralized optimization:** shared critic on global state (Eq. 38); GAE
  (Eqs. 46, 47); dual-clip surrogate (Eqs. 48, 49); KL estimate (Eq. 51); value loss
  (Eq. 53); total loss (Eq. 54) with AdamW/cosine LR; adaptive KL weight (Eq. 52).
- **Output:** trained decentralized policy `pi_theta`.

## Manuscript change

- **Added (blue):** Algorithm 1 after Eq. (26) in Section III-B, plus a one-line
  pointer sentence; Algorithm 2 after Fig. (ART-MAPPO workflow) in Section IV-D, plus
  a one-line pointer sentence.
- No equations were changed; the boxes only reference existing equations. Compilation
  is clean (18 pages), all equation cross-references in both boxes resolve correctly,
  and there are no unresolved references in the document.

*Note: no separate figure is required for this comment; the deliverable is the two
in-paper algorithm boxes themselves.*
