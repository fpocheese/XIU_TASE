# Reviewer 2 applicability revision

## Objective

Extend the current readability-revised manuscript for the reviewer request on communication
delay, sensing uncertainty, and three-dimensional dynamics.

## Required manuscript changes

1. Replace the planar point-mass model with a consistent 3-D model.
2. Extend ART-MAPPO action, observation, reward, metrics, and notation to include vertical
   motion, pitch, and `n_z`.
3. Replace the two-case/two-method result block with the v10 figures in
   `new_sim_fig/figures_v10`.
4. For every method/case show separate 3-D and horizontal trajectories, three independent
   single-column two-panel figures pairing $n_y$ with heading, $n_z$ with pitch, and $n_x$
   with velocity, followed by `t_go` and terminal time synchronization.
5. Keep every figure single-column and retain the original IEEE figure/caption style.
6. State delays in milliseconds, never in simulation steps, and place all delay/noise settings
   only in the simulation scenario-description subsection.
7. State observation-noise mean, variance, and physical units. Primitive position noise is
   zero-mean with variance 9 m^2 per coordinate; primitive velocity noise is zero-mean with
   variance 0.09 (m/s)^2 per component. Derived observations are computed from those delayed,
   noisy primitives rather than assigned unsupported independent variances.
8. Regenerate both clean and blue-highlighted manuscripts and compile both.

## Acceptance checks

- No residual claim that the model/action is planar.
- `a_i=[n_x,n_y,n_z]^T`; reward and terminal load include `n_z`.
- All 40 requested result plots are present (10 per case/method), together with one defender
  legend in each of the 12 two-panel command/response figures, for 52 image inclusions in total.
- No `figure*` is introduced in the revised simulation-result block.
- No simulation delay is reported as a number of steps, and its numerical setting appears only
  in the simulation setup.
- Clean and highlighted PDFs compile with resolved references and bibliography.
