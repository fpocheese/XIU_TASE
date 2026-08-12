# Validation report

## Scope and immutability

- The nominal assignment problem is **20 interceptors against 8 targets**.
- The experiment is assignment-only; no guidance, policy inference, or
  interception outcome is simulated.
- The highlighted manuscript and both original IDBO code bases were read-only.
- SHA-256 of `new_highlight/main.tex` after all experiments:
  `224f7c091217c2749d07fc19097a33ad43a68c81456fa0e0715bd1b0e9dfad5c`.
- V2 source hashes:
  - `idbo_paper.py`:
    `e0e2bf23320b44ed5c8f66458ea6365f02fa5b17c4e0cae4d1a60b354b21699f`
  - `scenario_paper.py`:
    `6b6d85ac909a13dec66306f8b0f5e1218ecff18cdf3c33deaadd61df0c7ecf3c`
- V1 source hashes:
  - `idbo.py`:
    `f5504b31ddb31b34bde2950b260f423d83d9d6c9e023a3a75994f669f24209cb`
  - `scenario.py`:
    `6dd3622ff5a6bcbfbacd0feeb61dc88e5a2c3fb45c74f64dd6db47bb58db3737`

## Runtime environment

- CPU: 12th Gen Intel Core i7-12700H
- Python: 3.10.12
- NumPy: 1.24.2
- Matplotlib: 3.7.1

CPU runtimes are wall-clock measurements on this host and should be interpreted
as implementation-specific scaling measurements, not hardware-independent
onboard timing guarantees.

## Numerical checks

- All seven Python files pass `python3 -m py_compile`.
- All V1/V2 CSV files parse successfully.
- Static-delay fixed point: 100/100 verified runs for each version.
- Topology fixed point: 60/60 verified runs for each version.
- Scale fixed point: 25/25 verified runs for each version.
- The paired V1/V2 file contains finite numerical data for all 60 rows
  (30 identical scenes per version).
- No unexpected missing value, NaN, or Inf was found. The only NaNs are the 22
  intentionally undefined per-seed recovery-time entries in
  `dynamic_run_summary.csv` when no changed epoch recovered. Their corresponding
  recovery rate is zero and they are excluded from recovery-time averaging.
- V1 and V2 dynamic traces each contain 48,000 communication-time samples.

## Plot and LaTeX checks

- Each PDF figure is a valid single-page vector file; PNG files were exported at
  600 dpi and SVG files are present.
- The V1, V2, and comparison PNG figures were visually inspected for empty axes,
  clipped labels, hidden legends, and inconsistent units. The nominal
  `20×8` point is explicitly marked in the scaling plots.
- `table_delay_dynamic.tex`, `table_scalability.tex`, and
  `table_v1_v2_comparison.tex` compile under `IEEEtran` without overfull boxes or
  undefined commands.
- `reviewer_response_3_2_v2.tex` and
  `manuscript_revision_suggestion.tex` compile in standalone validation wrappers.
- The manuscript suggestion is a separate file only; it has not been inserted
  into or used to overwrite `new_highlight/main.tex`.

## Interpretation guardrails

- The V2-only results directly answer Reviewer 3.2 and are the recommended
  evidence for the response.
- The V1/V2 comparison uses the same current three-dimensional scenes, objective,
  seeds, and budgets. The legacy V1 fixed scene is deliberately not used.
- The paired data do **not** support a claim that V2 dominates V1 on every
  metric. The comparison is therefore retained as an internal diagnostic and is
  reported without data selection or manual adjustment.
