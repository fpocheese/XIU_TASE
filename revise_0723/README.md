# 2026-07-23 readability revision

Source baseline: `../origin_paper/XIU_tase_paper/main.tex` (kept unchanged).

## Deliverables

- `main_revised.tex` / `main_revised.pdf`: clean revised manuscript.
- `main_revision_highlight.tex` / `main_revision_highlight.pdf`: revised manuscript with
  modified material highlighted in blue.
- `main_modular.tex`: maintainable source that inputs the four revision modules below.
- `section_iii_revised.tex`: revised target-assignment section.
- `section_iv_revised.tex`: revised ART-MAPPO section.
- `evaluation_metrics_revised.tex`: evaluation definitions moved to the simulation section.
- `notation_table_revised.tex`: grouped notation appendix.
- `simulation_cases_3d_revised.tex`: four single-column 3-D simulation-result blocks
  (two cases by two methods).

## Scope of the blue highlighting

The complete revised Sections III and IV, the relocated evaluation-criteria subsection, the
three renamed hyperparameter entries, and the grouped notation appendix are blue. Unchanged
material remains black. This block-level convention is used because formula-level deletion
markup can break `amsmath`, `algorithmic`, and table environments when entire derivations are
restructured.

## Readability outcome

- Section III: 22 displayed equations reduced to 9 core numbered equations.
- Section IV: 41 displayed equations reduced to 14 core numbered equations.
- Sections III--IV total: 63 reduced to 23, within the planned range of 20--24.
- Standard details are consolidated into prose, tables, and algorithm boxes.
- Physical meaning is stated after the core mappings, and unsupported claims/bounds are
  removed or weakened to their justified scope.
- Evaluation metrics now specify arrival time and terminal measurement window.
- The notation appendix is grouped into physical, assignment, learning, and evaluation terms.

Both manuscript PDFs were built successfully with `latexmk` using the original bibliography
and figure assets.

## Reviewer 2 applicability extension

- The planar kinematics were replaced by a three-dimensional point-mass model with
  $(x,y,z)$, heading $\gamma$, flight-path angle $\theta$, and the three command channels
  $(n_x,n_y,n_z)$.
- The assignment geometry, ART-MAPPO action/observation/reward definitions, terminal-load
  metric, and notation summary were made consistent with the 3-D model.
- The simulation setup now states the communication delay in milliseconds and reports the
  mean, variance, and units of the Cartesian position and velocity errors. These numerical
  delay/noise settings are confined to the setup subsection; derived observations are computed
  from the delayed noisy Cartesian states.
- The two cases and two methods use the v10 results. Each of the four blocks contains a 3-D
  trajectory, horizontal projection, three full-width two-panel command/response figures
  ($n_y$--heading, $n_z$--pitch, and $n_x$--velocity), a time-to-go plot, and a terminal
  synchronization plot. This gives 40 result plots plus 12 repeated legends, all in
  single-column `figure` environments.
- `main_revised.pdf` and `main_revision_highlight.pdf` both compile to 22 pages with resolved
  citations/references and no overfull boxes.
