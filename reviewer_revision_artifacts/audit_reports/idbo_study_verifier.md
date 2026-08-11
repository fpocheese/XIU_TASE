# Independent Verification of `idbo_reviewer_study.py`

Date: 2026-07-18

Scope: independent audit of
`revision_autoresearch/experiments/idbo_reviewer_study.py` and
`revision_autoresearch/results/idbo_reviewer_study`. The experiment script,
archived results, and manuscript were not modified.

## Executive verdict

The script is deterministic and internally correct for a **centralized synthetic
many-to-one assignment surrogate**. Its assignment representation, capacity repair,
survival-probability objective, elitist best-so-far history, and non-runtime summary
values passed independent checks.

It is **not an implementation of the distributed IDBO algorithm described in the
manuscript**. In particular:

1. The four named roles are simplified categorical mutation rules, not the manuscript's
   rolling, dancing, breeding, and stealing equations with adaptive
   \(\alpha,\beta,\gamma,\delta,\delta',\eta\).
2. The "consensus" simulator floods sets of known node IDs. It does not communicate
   bids, maintain top-\(L_{\max}\) winner lists, resolve assignment conflicts, enforce
   the row/lower-capacity constraints after communication, or couple consensus back to
   optimization.
3. The one-node-failure experiment only asks whether the remaining nodes learn the IDs
   of all other remaining nodes. It does not test target reassignment, target coverage,
   assignment cost, or guidance after an interceptor failure.

Therefore, the archived numbers can support a narrowly labeled surrogate experiment,
but cannot support claims of distributed assignment convergence, delayed top-\(K\)
auction consensus, partial-failure robustness, end-to-end assignment-guidance
performance, or onboard real-time feasibility.

## Reproduction record

Archived artifact hashes:

```text
343858c0a043fda8d87e3df4fa2919ae8001a06aec3b8b01a9c56f748b10f2a4  idbo_reviewer_study.py
b09e12497d9f73b2a2ab95d8d0954154e950ab9173ef8601d83a6176c1d4513d  idbo_reviewer_summary.json
f2d9a8cdda95a7350c0f6511454db224d78ed14a8a1f25425bda2cf775248834  idbo_reviewer_tables.csv
```

Independent full rerun:

```bash
python3 revision_autoresearch/experiments/idbo_reviewer_study.py \
  --output /tmp/idbo_verify_a --seeds 30
```

After deleting the two wall-clock fields from each scalability row, the archived and
rerun JSON files were byte-identical:

```text
c4add8683fea81ee14e71431c57fd958794ae305a15288579b7c3dfeec55d9df  archived non-runtime JSON
c4add8683fea81ee14e71431c57fd958794ae305a15288579b7c3dfeec55d9df  rerun non-runtime JSON
```

The wall-clock values did not reproduce exactly:

| Size | Archived mean (s) | Independent mean (s) | Change |
|---|---:|---:|---:|
| 20 x 8 | 0.1163 | 0.1828 | +57.1% |
| 40 x 16 | 0.2240 | 0.3480 | +55.4% |
| 80 x 32 | 0.5924 | 0.8734 | +47.4% |

The independent run used the currently shared host, identified as a 12th Gen Intel
Core i7-12700H. Host load was not controlled. The timing trend is repeatable in
direction, but the absolute numbers are not archival constants.

## Problem definition and feasibility

### What is correct

- A candidate has one target index for every interceptor, so a 20-interceptor problem
  has 20 categorical decisions (`idbo_reviewer_study.py:71-75,136-142`).
- For the study sizes, `max_load = ceil(M/N)` gives capacities 3 for 20/8, 40/16,
  and 80/32 (`idbo_reviewer_study.py:41-68`).
- Repair first fills empty targets and then drains overloaded targets, selecting each
  move by its immediate survival-cost delta (`idbo_reviewer_study.py:78-133`).
- The objective exactly computes
  \(\sum_j\prod_{i:a_i=j}(1-P_{ij})\), under the independent-failure model
  (`idbo_reviewer_study.py:71-75`).

Independent tests found:

```text
60/60 final adaptive/fixed assignments feasible at 20 x 8
0/3000 random-repair property failures over 20 x 8, 40 x 16, and 80 x 32
0 numerical difference from a direct objective calculation on a 4 x 2 example
```

These checks apply to the study sizes. `repair` has no explicit infeasibility guard for
general inputs such as \(M<N\); the present experiment never uses such inputs.

### Model limitation

The generated \(P_{ij}\) matrix is a bounded synthetic proxy. Its "ZEM" term is
`distance * sin(heading_error)` rather than a propagated zero-effort miss based on
relative position and relative velocity (`idbo_reviewer_study.py:43-67`). It also
omits the manuscript's adversarial advantage, local preference fitness, neighbor
overload estimates, and velocity-dependent update. Results must therefore be labeled
as a fixed-snapshot synthetic probability study, not as validation of the full
engagement model.

## Optimizer and adaptive-schedule result

The best-so-far history is genuinely monotone because a candidate is accepted only if
it does not worsen its individual cost, while the recorded global incumbent changes
only under a strict improvement (`idbo_reviewer_study.py:187-205`). Across the 60
adaptive/fixed histories, the verifier found zero monotonicity violations.

The archived final statistics reproduce exactly:

| Mutation-size schedule | Final cost, mean +/- std | Median | Median iterations to within 1% of final |
|---|---:|---:|---:|
| Decaying 0.36 to 0.04 | 0.006975 +/- 0.000141 | 0.006921 | 19.0 |
| Fixed 0.18 | 0.006942 +/- 0.000083 | 0.006921 | 21.5 |

The paired 30-seed audit gives:

```text
adaptive better / equal / worse than fixed: 1 / 25 / 4
mean(adaptive - fixed): +3.3504e-05
median paired difference: 0
number of distinct final costs: 3 for each schedule
```

Thus the decaying mutation-size schedule reaches its own final 1% band 2.5 iterations
earlier in the median, while the fixed schedule has the slightly lower final mean.
There is no evidence here that adaptation improves final solution quality.

The result key and CSV label `coefficient_ablation` are misleading. The only quantity
ablated is a common mutation fraction (`strength`) at
`idbo_reviewer_study.py:187-196`. The experiment does not ablate any role, and does
not implement or vary the manuscript's six adaptive coefficients.

**Usable wording:** "On the fixed synthetic 20 x 8 snapshot, the decaying mutation-size
schedule reached a 1% terminal-cost band in 19 median iterations versus 21.5 for a
fixed mutation fraction; their median final costs were identical."

**Unusable wording:** "Adaptive IDBO coefficients significantly improve convergence
and final assignment quality."

## Weight sensitivity

The \(w_{\rm ZEM}=\{0.3,0.5,0.7\}\) means and standard deviations reproduce exactly.
The same geometry seed and corresponding optimizer seeds are used at all three
weights (`idbo_reviewer_study.py:288-295`), which is appropriate for a paired synthetic
sweep.

However, changing the weight changes every \(P_{ij}\) and therefore changes the
objective being reported. The raw optimized costs at different weights are not
measurements on a common evaluation criterion. The script also does not report
assignment-change rates or downstream interception metrics.

**Conditionally usable:** the three rows may be presented as a synthetic input-output
sensitivity sanity check, explicitly naming the proxy probability construction and
avoiding "better/worse" comparisons across weights.

**Not usable:** these values do not establish robustness of the physical block
probability model, nor do they show that one weight provides better interception
performance.

## Scalability statistics

The deterministic normalized-cost values reproduce. Each size uses 10 separately
generated synthetic problems, 60 iterations, and population 30
(`idbo_reviewer_study.py:297-317`). The target/interceptor ratio is held near 2.5.

Limitations:

- The random geometries are regenerated at each size and are not nested matched
  instances.
- Only the mean normalized cost is saved; its across-instance standard deviation is
  omitted.
- The runtime excludes distributed message handling because no assignment-consensus
  algorithm is run.
- No hardware metadata, process-load control, warmup protocol, or algorithmic
  complexity fit is stored.

**Usable with qualification:** the implementation's mean runtime increased
monotonically from 20 x 8 through 80 x 32 on both executions.

**Not usable:** the absolute archived times cannot establish asymptotic complexity,
distributed scalability, communication overhead, onboard real-time operation, or
Jetson-class deployment feasibility.

## Consensus-delay model

`consensus_rounds` constructs a fixed undirected ring-plus-jump graph and repeatedly
floods each active node's growing set of known node IDs
(`idbo_reviewer_study.py:209-248`). Delay is a uniform integer jitter from zero to the
specified maximum; each transmission is independently dropped; every active node
transmits its full current set in every round.

The archived 30-seed values are correct for that model. Under 1% dropout, all trials
completed and the mean dissemination rounds increased from 4.0 at zero extra delay to
5.8, 7.07, and 9.17 at maximum jitters of 1, 2, and 4 rounds. Independent spot checks
reproduced the exact sequences.

This result is only a **node-information flooding latency** experiment:

- `problem.probability`, assignments, costs, target indices, bids, and capacities are
  never read by `consensus_rounds`;
- the payload is an unbounded Python set rather than a specified winner-list message;
- there is no deterministic tie-breaking or stale-bid handling;
- there is no assignment disagreement metric;
- there is no feedback from dissemination to the optimizer.

**Usable wording:** "In a separate 20-node ring-plus-jump flooding model, bounded link
jitter increased full-information dissemination latency from 4.0 to 9.17 mean rounds
at 1% independent packet dropout."

**Unusable wording:** "The distributed IDBO assignment converged in 4-9 rounds under
delay and packet loss."

## Single-node failure

The reported 100% success and 7.17 mean rounds at delay 2 and dropout 1% are correct
for dissemination among the 19 surviving nodes. Each seed removes one node before the
flooding process starts (`idbo_reviewer_study.py:341-362`).

The test does not check whether the removed interceptor's target remains covered,
whether its assignment is reallocated, whether capacity constraints still hold, or
whether downstream guidance succeeds. Since success is defined only as all survivors
knowing the identities of all survivors, this number must not be described as
assignment or mission robustness to partial failure.

## Representative assignment

The archived representative assignment is feasible for the generated synthetic
problem, with loads `[3, 2, 2, 3, 2, 3, 2, 3]`.

It is not the paper/guidance topology in which targets 1-4 receive three defenders and
targets 5-8 receive two. It therefore cannot be used as the asserted end-to-end IDBO
input to the guidance simulations without rerunning guidance with this exact mapping
and archiving that mapping in the episode data.

## Statistical reporting audit

- Sample standard deviations use `ddof=1` correctly
  (`idbo_reviewer_study.py:251-259,330-337`).
- The adaptive/fixed and weight sweeps use common seed indices, but no paired
  confidence interval or hypothesis test is reported.
- The adaptive/fixed outcomes have heavy ties, so a generic normal-based significance
  claim would be inappropriate without a prespecified paired method.
- The plotted left panel correctly shows mean +/- one sample standard deviation.
- The plotted right panel correctly selects the 1% dropout rows, but should be labeled
  "information dissemination" rather than "consensus" if retained.
- The summary does not store raw per-seed outcomes, configuration hashes, platform
  metadata, message counts, or confidence intervals. This limits independent
  statistical analysis even though deterministic reruns recover the same values.

## Final claim matrix

| Result | Numerical validity | Manuscript use |
|---|---|---|
| 20 x 8 objective and feasible loads | Verified | Use for the synthetic centralized surrogate only |
| Monotone best-so-far histories | Verified | Use as an empirical/algorithmic elitism property, not global convergence |
| 19.0 vs 21.5 iterations | Verified | Use as a mutation-size schedule comparison |
| Final adaptive vs fixed cost | Verified | Report neutrally; fixed mean is slightly lower and 25/30 pairs tie |
| Probability-weight rows | Verified | Proxy sensitivity sanity check only |
| Normalized scaling costs | Verified | Synthetic instance results only |
| Absolute runtime values | Environment-dependent | Do not use for onboard/real-time claims |
| Delay/dropout rounds | Verified for set flooding | Do not call distributed assignment consensus |
| One-node removal | Verified for survivor set flooding | Do not call assignment, guidance, or mission failure robustness |
| Representative assignment | Feasible synthetic solution | Do not identify it as the guidance experiment topology |

## Required corrections before strong reviewer claims

To support the manuscript's current distributed-IDBO language, the experiment must
replace set flooding with the actual per-target bid/winner-list protocol, record
assignment feasibility and disagreement after every round, feed the consensus
assignment back into optimization, and evaluate delayed/stale messages. The operator
study must implement and separately vary the manuscript coefficients or revise the
manuscript to match the simplified mutation rules. Partial-failure evaluation must
measure reallocation, target coverage, assignment cost, and downstream interception.

Until then, the safest use of this study is a clearly disclosed, centralized synthetic
surrogate diagnostic plus a separate communication-flooding illustration.
