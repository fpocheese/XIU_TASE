# Re2_4 — Environmental modeling beyond UAV dynamics

**Reviewer comment.** "The simulation environment is rather simplistic, and apart from UAV
dynamics, there is no other environmental modeling, which does not align with the high-fidelity
simulation environment claimed in the paper."

## Response

The revised environment models several effects beyond the bare UAV dynamics, and the whole
campaign is re-run in three dimensions:

1. **Communication delay.** State/velocity packets are delivered after τ_s = 100 ms; command
   packets after 50 ms with a first-order actuation response.
2. **Sensing uncertainty.** Zero-mean Gaussian error on each Cartesian position (σ = 3 m) and
   velocity (σ = 0.3 m/s) component; all derived geometry (range, altitude difference, LOS
   direction, closing speed, heading, pitch, velocity-to-LOS angle, t_go) is recomputed from the
   delayed, noisy states.
3. **Communication packet loss.** 1% command-packet dropout handled by a zero-order hold.
4. **Three-dimensional engagement geometry.** Defenders start low, attackers start elevated, so
   every engagement demands coupled horizontal + vertical closure.
5. **Distributed / networked execution.** Rather than one monolithic simulator, five interceptor
   policy endpoints run on separate embedded computers and exchange packets with the simulator
   over the network under the above channel — a hardware-in-the-loop (HIL) semi-physical setup
   (see Re3_10).

These settings are collected in a new equation (noise model) and a **Communication Delay and
Observation-Uncertainty Settings** table, and the physical realism is exercised end-to-end by the
HIL loop. This substantiates the "high-fidelity" description: the environment now includes
transport delay, stochastic sensing, packet loss, 3-D coupling, and real inter-node communication.

## Where the manuscript changed
- New environmental-modeling paragraph + noise-model equation + delay/uncertainty table in the
  experimental-setup subsection (Section on Simulation Verification).
- HIL semi-physical description (shared with Re3_10).
- 3-D initialization column added to the UAV-parameter table.
