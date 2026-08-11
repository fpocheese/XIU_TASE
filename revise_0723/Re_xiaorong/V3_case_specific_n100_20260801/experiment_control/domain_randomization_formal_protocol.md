# Formal eight-model case-specific ablation protocol (domain-randomized reset)

- Eight fresh models: four variants independently trained in each of Case 1
  and Case 2; one common training seed (`8303`) and one common initial-state
  stream per case for controlled ablation comparison.
- Budget: 600,000 requested / 599,040 realized environment steps per model.
- Initial-state perturbation scale 1.0, exactly matching the evaluator's
  physical position/heading/speed distribution.  No sensor noise, delay, or
  actuator lag is added.
- Case 1: synchronization gain 1.40, residual scale 0.20.
- Case 2: synchronization gain 2.00, residual scale 0.05.
- Both cases use the paper's mean time-to-go group reference.
- Selection seeds: Case 1 96001--96020; Case 2 96201--96220.
- Final paired tests: exactly 100 episodes per variant/case; Case 1 seeds
  98401--98500 and Case 2 seeds 99401--99500.
- No final-test seed is used for training, calibration, or checkpoint selection.
- Original source tree, original trained weights, and original `main.tex` are
  not modified.
