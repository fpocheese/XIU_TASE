# Reviewer 2 revision validation

Date: 2026-07-23

## Structural checks

- Clean source is generated from `main_modular.tex` with `latexpand`.
- The modeling equations propagate three-dimensional position and velocity.
- ART-MAPPO uses `a_i=[n_x,n_y,n_z]^T`; its observation is 17-dimensional and includes
  vertical/LOS information and prior three-channel action.
- Pairwise assignment alignment uses the three-dimensional velocity-to-LOS angle.
- Terminal load uses `sqrt(n_y^2+n_z^2)`.
- The result subsection contains 28 single-column figure environments and no double-column
  figure environment.
- The result subsection includes 52 images: 40 requested result plots and 12 legends.

## Delay and uncertainty checks

- The scenario setup reports a fixed 50 ms end-to-end delay, never a step count.
- Position-component error: mean 0 m, variance 9 m^2.
- Velocity-component error: mean 0 m/s, variance 0.09 (m/s)^2.
- Derived observation quantities are recomputed from the delayed noisy Cartesian states rather
  than being assigned unsupported independent Gaussian parameters.
- The 300 ms value is identified as the command-response time constant, not an observation
  delay.
- Numerical noise/delay settings are stated only in the simulation setup.

## Build and visual checks

- `main_revised.pdf`: 22 pages; references and bibliography resolved; no overfull boxes.
- `main_revision_highlight.pdf`: 22 pages; references and bibliography resolved; no overfull
  boxes.
- Pages 9 and 11--20 were rendered and inspected. The uncertainty table fits one column, and
  the four method/case result blocks preserve readable single-column 3-D, plan-view, three
  full-width two-panel command/response, time-to-go, and time-synchronization figures.
