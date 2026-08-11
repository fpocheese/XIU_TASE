# Reviewer 1 readability audit

## Scope

- Source of record for analysis: `origin_paper/XIU_tase_paper/main.tex`.
- Sections III and IV: explain every displayed equation and recommend keep, merge,
  move, simplify, or delete; do not edit the paper in this task.
- Notation: audit definitions, reuse, conflicts, and avoidable symbols across the paper;
  do not edit the paper in this task.
- Figures: preserve the existing v10 outputs and source; implement a workspace-local
  v11 plotting script and generate candidate de-cluttered figures for comparison.

## Acceptance checks

1. Every labeled displayed equation in Sections III and IV appears in the audit.
2. Recommendations distinguish presentation density from technical correctness.
3. Symbol findings cite locations and propose a consistent replacement scheme.
4. v11 reads the same exported simulation data, does not modify v10, and produces
   reproducible candidate images.
5. Candidate images are visually inspected at publication-like size.

## Decisions

- Treat `origin_paper/XIU_tase_paper/main.tex` as the requested original manuscript,
  even though a separately revised `main.tex` exists at the workspace root.
- The discovered v10 source is read-only outside the workspace; v11 will be created
  inside this workspace and will leave that file and all `figures_v10` directories intact.
