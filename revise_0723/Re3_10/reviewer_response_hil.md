# Re3_10 — Hardware-in-the-loop / semi-physical validation

**Reviewer comment.** "At last, it is advisable to incorporate hardware-in-the-loop and
semi-physical simulation validation to improve the practical credibility of the ART-MAPPO
cooperative interception strategy for realistic autonomous UAV defence applications effectively."

## Response

The three-dimensional experiment is executed as a hardware-in-the-loop (HIL) semi-physical study:

- **Embedded interceptor endpoints.** Five interceptor policy nodes (D0–D4) run on separate
  NVIDIA Jetson NX embedded computers. Each node holds its own recurrent (GRU) state, an
  observation-delay line, and an actuator-delay line — exactly as a physical interceptor would.
- **Mathematical simulation for the rest.** The remaining interceptors and all attackers evolve
  in the simulator, which exchanges state/command packets with the NX nodes over the network.
- **Realistic channel.** The exchange runs under the delay/noise/packet-loss channel: sensor
  delay 2 steps = 100 ms, actuator delay 1 step = 50 ms, normalized observation-noise σ = 0.003,
  normalized action-noise σ = 0.02, command dropout 1%, at a 0.05 s simulation step.

This closed loop validates the decentralized-execution assumption of the CTDE design on real
processors and real inter-node communication, rather than in a single monolithic process, which
is precisely the practical-credibility gap the reviewer raises. Both cases succeed under HIL:
ART-MAPPO holds the terminal arrival spread to 0.25 s (Case 1) and 0.30 s (Case 2); the
capacity-matched PN reference gives 0.40 s and 0.35 s respectively.

## Where the manuscript changed
- New **"Hardware-in-the-Loop Semi-Physical Validation"** paragraph in the experimental-setup
  subsection describing the 5-NX split-execution architecture and the channel.
- New **platform schematic** (`HIL_fig/tase_HIL.png`, drawn by the author) added as a
  full-width `figure*` (auto-numbered Fig. 7). The server side hosts the engagement-state
  update, the attacker cluster, and the software policies of D5–Dn; the five Jetson NX nodes
  run the hardware policy endpoints D0–D4 (online inference + target assignment), connected
  over a physical TCP Ethernet link (TP-LINK switch). The caption reconciles the figure's
  agent index A_k with the manuscript's interceptor label D_k (A_k = policy of D_k).
- The results figures reflect the HIL runs; the robustness subsection (Re3_7) and the KL
  overhead analysis (Re3_6) both build on this deployed configuration.

## Where the response letter changed
- The same schematic is reproduced as **Fig. R13** inside Comment 3.10, introduced by a
  short `\evi` block; the letter `\graphicspath` now includes `../HIL_fig/`.

## Note
The reviewer's phrase "cooperative tapping strategy" is read as "cooperative interception
strategy" (the paper's topic); the response is written accordingly.
