# Re2_5 — Episode horizon vs. step size, and the D1/D2/D3 setting

**Reviewer comment.** "In Section V.C, the authors state that each episode has at most 25
steps during training, while in Section V.A the simulation step size is set to 0.05s. This
implies a training period of 1.25s, which is impossible to learn any interception strategies
given an engagement distance of 1500 meters and a maximum speed of 40 m/s. By the way, the
settings 'D1 = 100 m, D2 = 1500 m, and D3 = 1600 m' are inconsistent with the illustrations
in Fig. 6."

## Response (author's prescribed wording)

The maximum of 2500 steps per episode has been revised in the main text. The original
setting "D1=100m, D2=1500m, D3=1600m" was incorrect and has been revised to: D1=1500m,
D2=1600m, D3=100m.

## Why this resolves both points

- **Horizon.** The "25" was a transcription typo for 2500. With the 0.05 s step, 2500 steps
  give a 125 s engagement window — ample to close a 1500 m separation at 40 m/s (nominal
  closing time ≈ 1500/40 = 37.5 s, well inside 125 s). The "impossible in 1.25 s" objection
  no longer applies.
- **Geometry / Fig. 6.** Corrected to D1=1500 m, D2=1600 m, D3=100 m. Attackers spawn in the
  far annulus r ∈ [D1, D2] = [1500, 1600] m and descend toward the defended center; defenders
  start clustered within a radius-D3 = 100 m disk around the center. This matches the
  initial-geometry illustration (Fig. 6 in the reviewer's numbering = `chushi01.pdf`,
  `\label{chushi}` in the manuscript): far attacker annulus, central defender cluster.

## Where the manuscript changed
- **Section V.C:** episode horizon `\hldel{25}\hladd{2500}` steps per episode.
- **Section V.A:** representative-scenario setting corrected —
  D1 `\hldel{100}\hladd{1500}`, D2 `\hldel{1500}\hladd{1600}`, D3 `\hldel{1600}\hladd{100}`,
  plus a blue clause tying the values to the initial geometry (far annulus D1–D2 attackers,
  radius-D3 central defenders, consistent with `Fig.~\ref{chushi}`).

## Where the response letter changed
- New **Comment 2.5** entry (after Comment 2.4, before Reviewer 3): verbatim reviewer box,
  `\rsp` conceding both transcription errors with the author's wording, `\loc` pointing to
  Section V.C (horizon) and Section V.A (D1/D2/D3 + geometry figure).
