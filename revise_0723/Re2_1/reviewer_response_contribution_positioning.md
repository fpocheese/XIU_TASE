# Re2_1 — Are reward shaping and dual-clip/adaptive-KL generic RL techniques, or a genuine contribution?

**Reviewer comment.** "The author lists 'introducing a reinforcement learning-based reward
function to achieve precise time coordination' as the third core contribution. However, in the
Multi-Agent Reinforcement Learning (MARL) field, adding consensus-based penalty terms to the
reward function (Equation 64) to guide agents toward consistency is a rather common reward-shaping
technique. 'Dual clipping with adaptive KL' is a common method used in RL training to prevent
gradient explosion, yet the author packages it as a core innovation for solving 'aircraft overload
saturation'."

## Response

We fully agree with the reviewer on both technical points and thank them for pushing us to state
the contribution more precisely. A consensus penalty in a reward is a standard reward-shaping
device, and dual-clip together with adaptive-KL is a standard policy-gradient stabilizer — we did
not invent either mechanism and no longer claim to. The third contribution was over-stated as if
the *mechanisms* were novel; that wording was misleading and has been corrected.

Our actual claim is narrower: the contribution is not the two generic mechanisms but their
**problem-specific instantiation** for cooperative interception of a maneuvering swarm — the
particular *form* each takes, *why* that form matches this engagement, and the *measured* benefit
here. We do not refute the reviewer; we sharpen the claim.

### (1) The t_go-consensus reward is an impact-time (salvo) constraint, not generic "agreement"
Generic consensus shaping drives agents toward *any* shared value. Here the shaped quantity is
deliberately the time-to-go, and the target is not "be consistent" but "strike within one lethal
window": Eq. (68) penalizes |t_go_i − mean t_go| so a distributed salvo becomes simultaneous. This
matters because the tactical failure mode of a maneuvering swarm is exactly **sequential,
piecemeal** interception — interceptors spread out in time are defeated one at a time and the
swarm's evasive maneuver regenerates the geometry between arrivals. Classical cooperative
interception addresses this with analytic impact-time / salvo guidance laws; what is new here is
that the *same* tactical objective is carried by a **learned** reward term inside a MARL policy, so
simultaneity is achieved jointly with evasive-target tracking and target assignment, rather than by
a hand-tuned guidance law that assumes a cooperative or non-maneuvering target.

- Measured: temporal-coordination error E_co-time < 0.10 s (near-simultaneous salvo).
- Illustration (Fig. R12a): six interceptors starting with a 3.49 s time-to-go spread contract to
  a common impact instant (< 0.10 s dispersion).

### (2) Dual-clip here is a load-factor saturation guard, not a generic gradient stabilizer
The reviewer is right that dual-clip/adaptive-KL is a standard cure for gradient explosion. The
point we should have made is **what the bounded object physically is**. In interception, the
negative-advantage samples *are* the aggressive terminal-overload commands — the actions that spike
the commanded normal load factor n_y as the interceptor closes on an evading target. Under the
single clip, Â < 0 leaves the ratio r_t unbounded (Eq. 52), so one off-policy negative-advantage
mini-batch can drive an arbitrarily large commanded-overload increment; on an airframe that is a
hard actuator/structural saturation, not just a noisy gradient. Confining r_t ∈ [1/c, c]
(Eqs. 53–54) therefore bounds the per-update commanded-overload increment Δn_y — a **load-factor
saturation guard** — and the adaptive-KL band (Eq. 56) keeps successive policies inside the safe
envelope between updates.

- Illustration (Fig. R12b): for negative-advantage samples the single clip leaves the commanded-
  overload increment unbounded; the dual-clip floor caps it.

### (3) The instantiation pays off measurably here
The two problem-specific choices produce the manuscript's reported gains:
- Terminal overload held at **E_n ≈ 0.20 g** in the high-maneuver Case 2 at a **97.14%**
  interception success rate, whereas baseline PN saturates at **E_n → 1.0 g** and its success rate
  collapses to **17.5%**.
- Cross-seed reward std **halved** (0.64 vs 1.34), and 90%-optimality reached ~2000 episodes
  earlier (5591 vs 7654).
- E_co-time < 0.10 s (salvo timing).

These are summarized in the letter's Table R-2.1 and Fig. R12c (measured payoff vs. baseline).

### Related responses
This comment is closely tied to:
- **Comment 3.4** — theoretical comparison of these mechanisms vs. conventional MAPPO variants.
- **Comment 3.5** — reward-weight sensitivity study (each term governs its intended tactical axis).
- **Comment 3.6** — negligible real-time cost of dual-clip/adaptive-KL (training-loss only; 0 ms at
  execution).

## Where the manuscript changed
- Section I (Contributions), third contribution: the overstated wording is struck out (red) and
  replaced (blue) to concede the mechanisms are standard while scoping the contribution to their
  problem-specific instantiation (salvo impact-time reward; load-factor saturation guard). The
  bodies of Section IV-E (reward, Eqs. 64/68) and Section IV-D (dual-clip/KL, Eqs. 52–56) already
  describe both mechanisms in these interception-specific terms; only the contribution statement is
  revised.

## Note on honesty of the figure
All three panels are computed **directly from the manuscript's own equations and already-reported
metrics** — no new experimental data. Panel (a) integrates the t_go-consensus contraction of
Eq. (68); panel (b) evaluates the dual-clip ratio bound of Eqs. (53)–(54); panel (c) plots metrics
already reported in the manuscript against the baseline.

## Reproduce
```
python3 code/positioning_re2_1.py   # -> positioning_re2_1.pdf/.png
```
