# Re1_5 — Real-world applicability: delays, sensing uncertainty, 3D dynamics

**Reviewer comment.** "Expand discussion on real-world applicability, including communication
delays, sensing uncertainty, and 3D dynamics."

## Response

The revision addresses all three items directly in the manuscript, and the entire simulation
campaign has been re-run in three dimensions with these effects present in the loop.

### 1) 3D dynamics
The point-mass model is generalized from the planar form to full three-dimensional motion:
- Kinematics: ẋ = V cosθ cosγ, ẏ = V cosθ sinγ, ż = V sinθ (adds flight-path/pitch angle θ).
- Load channels: V̇ = nₓg, γ̇ = n_yg/(V cosθ), θ̇ = n_zg/V (adds the vertical normal load n_z).
- Relative geometry and ZEM are rewritten in vector form (r_AD, LOS unit vector, ‖r_AD + (v_A−v_D)t_go‖).

The three-dimensional variable definitions follow the pitch-plane convention used in the V10
plotting/evaluation code (pitch θ = arctan2(v_z, ‖(v_x,v_y)‖)), so the model and the figures are
mutually consistent. Every Case-1/Case-2 result figure is replaced with its 3D version: a 3-D
trajectory, a horizontal projection, and separate yaw-plane (n_y, γ), pitch-plane (n_z, θ), and
axial (n_x, V) responses, for both ART-MAPPO and the capacity-matched PN reference.

### 2) Communication delays
Every defender receives its state/velocity packets after a finite transport delay
(τ_s = 100 ms sensing; 50 ms command channel), and the derived geometry and t_go are recomputed
from the delayed states. This is documented in the new environmental-modeling paragraph and the
delay/uncertainty table.

### 3) Sensing uncertainty
Zero-mean Gaussian error is injected on every Cartesian position (σ = 3 m) and velocity
(σ = 0.3 m/s) component; the guidance observation absorbs it through the LOS-rate jump-rejection
filter. The new robustness subsection quantifies a ~6× noise attenuation on the guidance-critical
channel (see Re3_7).

## Where the manuscript changed
- **Section II (Problem Modeling):** 2D kinematics/LOS/ZEM struck and replaced with 3D forms
  (three new blue equations + updated symbol definitions).
- **Section on Simulation Verification:** new environmental-modeling paragraph + delay/uncertainty
  table; all Case-1/Case-2 figures rewired to the 3D V10 results; new "Real-World Applicability
  and Robustness" subsection.
- **Conclusion:** future-work list updated (communication delay now addressed; aerodynamic
  coupling added).
