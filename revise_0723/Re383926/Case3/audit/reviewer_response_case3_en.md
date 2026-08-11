**Additional end-to-end generalization experiment (Case 3).**  
To further examine whether the proposed assignment-and-guidance pipeline can
operate beyond the two scenarios used in the main evaluation, we added a third,
previously unseen engagement condition. Case 3 differs from Cases 1 and 2 in
both initial geometry and attacker policy: the eight attackers occupy a wider
nonuniform shell with staggered altitudes, heterogeneous speeds, and biased
inbound headings, and execute a three-stage multi-sine, bang-bang, and
frequency-chirped maneuver with vertical excitation. All perturbations are
seeded and reproducible.

For every trial, IDBO receives the actual transformed 3-D positions and
velocities and computes a new 20-to-8 assignment before guidance begins. The
recurrent Case-2 policy is then evaluated deterministically with frozen
parameters; no retraining, gradient update, or checkpoint selection is
performed on the 100 formal seeds. The validation seeds used to fix the
guidance configuration are disjoint from formal seeds 74001--74100. The learned
residual remains active (scale 0.20), so the reported result is not a
guidance-only baseline.

Across 100 Monte Carlo trials, all 2,000 defender trajectories reached their
assigned lethal radius and all eight attacking vehicles were intercepted in
every trial. Thus, the individual-interceptor, target-coverage, and all-target
interception rates were all 100%. Seventy-two trials also satisfied the strict
0.5-s coordination requirement for every assigned target group, giving a
complete cooperative-mission success rate of 72.0% (Wilson 95% confidence
interval: approximately 62.5%--79.9%). Importantly, the remaining 28 trials
were coordination failures rather than interception failures: each still
achieved 20/20 hits, but at least one group exceeded the synchronization
tolerance. Their worst-group arrival spread had a median of 1.20 s.

Following the definitions in the revised manuscript, the terminal metrics were
computed only over the 72 completely successful trials. The mean values were
\(E_{\mathrm{co-time}}=0.02678\) s,
\(E_n=0.79570\) g,
\(E_{\mathrm{miss}}=1.52028\) m, and
\(E_t=35.8480\) s. The corresponding standard deviations were 0.01145 s,
0.07579 g, 0.21625 m, and 0.17070 s. IDBO returned a feasible assignment in all
100 trials without invoking the deterministic repair; its mean one-shot
runtime was 805.7 ms.

These results provide a deliberately conservative generalization statement.
The frozen policy preserves complete target interception under the new
geometry and unseen hybrid maneuver, while the reduction to 72% under the
stricter all-group temporal criterion exposes the remaining limitation in
zero-shot synchronization. We have therefore not characterized Case 3 as
perfect generalization; instead, it demonstrates both the robustness of the
end-to-end interception chain and the specific coordination margin that can be
improved by future scenario-specific adaptation.

