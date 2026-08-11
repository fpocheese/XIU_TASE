# Delivery validation

The package was validated after relocation to `Re_xiaorong`.

## Formal data validation

- status: passed;
- training metric files: 40;
- training update rows: 8,000;
- evaluation episode files: 40;
- held-out evaluation episodes: 800;
- required-field NaN/Inf or structural errors: 0.

## Artifact coverage

- evaluation figures (PNG/PDF/SVG): 320;
- evaluation CSV/NPZ data files: 200;
- final combined figure: PDF, SVG, and 600-dpi PNG;
- final actor, critic, and restartable checkpoints: retained for every run;
- training and evaluation logs: retained;
- exact source snapshot and experiment-control scripts: retained.

All PNG, PDF, and SVG files were checked for the expected file signature. The
final combined PNG was visually inspected after relocation: all four panels
are populated, legends do not cover the principal curves, axes and labels are
visible, and confidence intervals are rendered.

Three zero-byte TensorBoard event files are retained in their original form.
They are empty startup artifacts from three Case-1 seed-704 launches, not
missing experimental records; the corresponding `training_metrics.csv`,
manifests, summaries, checkpoints, and evaluation files are present and pass
the formal validator.

`POST_TRANSFER_VALIDATION.json` records the machine-readable validation
result. `DELIVERY_FILE_LIST.txt` enumerates the package, and
`DELIVERY_SHA256SUMS.txt` contains delivery-level SHA-256 hashes.
