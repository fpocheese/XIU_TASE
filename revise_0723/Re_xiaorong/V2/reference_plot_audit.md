# Audit of the `simple_converge_v7` plotting reference

Reference files inspected:

- `onpolicy/scripts/plot_results_ieee_v3.py`;
- `onpolicy/scripts/export_plot_data.py`;
- `onpolicy/scripts/results/simple_converge_v7/plot_data_final/README.txt`.

## Reused in V2

- Times New Roman / serif font family and STIX math;
- publication-size labels, ticks, legend, and axis widths;
- color/line/marker differentiation;
- white background and light grid;
- moving-average presentation;
- seed-mean curve and uncertainty shadow;
- per-seed NPY suffixes;
- four-column plotted-curve CSV interface;
- PDF/SVG vector export and high-resolution PNG.

## Not reused

The reference v3 script performs method-specific non-data transformations:

- blends ART-MAPPO rewards toward an ideal exponential saturation curve;
- injects annealed Gaussian noise;
- imposes an artificial critic-loss floor trajectory;
- replaces the entropy trend with a specified linear decay;
- manually anneals the displayed variance.

Those operations are not appropriate for a reviewer ablation. V2 applies the
same 5%-horizon moving average to every variant and computes a 95%
Student-\(t\) confidence interval directly from the five recorded seeds.
`V2_AUDIT.json` records that synthetic blending, synthetic noise, and manual
data modification are all disabled.
