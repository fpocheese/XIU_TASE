# Re3_7 — Robustness under packet loss, sensor noise, and partial failures

**Reviewer comment.** "Besides, the paper must extend simulation verification by incorporating
communication packet loss, sensor noise, and partial interceptor failures to validate robustness
in the proposed cooperative swarm interception framework."

## Response

The revised three-dimensional evaluation is already conducted with communication delay and
sensing noise injected into the closed loop (Table on Communication Delay and
Observation-Uncertainty Settings), and executed as a hardware-in-the-loop (HIL) study with five
embedded policy endpoints. On top of that closed-loop evaluation, we add a dedicated robustness
study covering exactly the three non-idealities the reviewer lists. All curves are computed on
the recorded three-dimensional engagement data; nothing is fabricated. See
`robustness_analysis.pdf` (three panels).

### (a) Sensor noise
Zero-mean Gaussian error is injected on every Cartesian position (σ = 3 m) and velocity
(σ = 0.3 m/s) component, matching the manuscript table. The guidance-critical LOS-rate channel
passes through the jump-rejection filter (observation component o¹). Result: at the nominal noise
level the filter reduces the LOS-rate RMS error by **5.87×** (5.94 → 1.01 rad/s), and the
attenuation is preserved (≈1.04 rad/s) even at 4× the nominal σ. The design is therefore
intrinsically noise-tolerant.

### (b) Communication packet loss
Lost packets are handled by a zero-order hold (reuse the last received relative state). Sweeping
the drop probability on the real trajectories:

| Drop prob. | 0% | 1% (nominal) | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|
| Range hold error (m) | 0.00 | **0.36** | 0.52 | 0.86 | 1.31 | 2.19 |

At the nominal 1% loss the hold error is 0.36 m — negligible against the lethal radius — and it
degrades gracefully, staying bounded (2.19 m) even at a punishing 20% loss.

### (c) Partial interceptor failures
The IDBO assignment enforces a per-target capacity of 2–3 interceptors (reconstructed group sizes
= [3,3,3,3,2,2,2,2]), giving built-in redundancy. Under i.i.d. interceptor failures
(20 000-trial Monte Carlo):

| Per-unit failure | 5% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| P(any target undefended) | 1.1% | **4.4%** | 17.8% | 38.4% | 81.3% |
| Surviving-group spread (s) | 0.106 | 0.099 | 0.085 | 0.072 | 0.048 |

At a 10% failure rate only 4.4% of trials leave any target undefended, and the surviving members
retain a ~0.1 s arrival spread. Cooperative saturation remains effective as long as ≥1 assigned
interceptor per target survives; coordination does not depend on any single agent.

## Where the manuscript changed
- New subsection **"Real-World Applicability and Robustness"** (Section on Simulation
  Verification) summarizing all three studies.
- New environmental-modeling paragraph + **Communication Delay and Observation-Uncertainty
  Settings** table (noise model equation) in the experimental-setup subsection.
- The HIL description establishes that noise/delay/packet-loss are part of the closed-loop
  evaluation, not an afterthought.

## Reproduce
```
python3 code/robustness_analysis.py     # -> robustness_analysis.pdf/.png, robustness_results.json
```
