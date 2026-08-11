"""
Lightweight script to (re)generate plots from saved experiment data.

Usage:
    python plot_from_data.py --infile results/results.pkl --outdir results

This script loads the pickled `results` dict produced by `main.py` and
recreates the `convergence` and `violin` plots without titles (suitable
for paper panel placement).
"""
import os
import argparse
import pickle

# import plotting functions from main (they won't execute main())
import main


def main_cmd():
    parser = argparse.ArgumentParser(description='Plot from saved results')
    parser.add_argument('--infile', type=str, default='results/results.pkl',
                        help='Path to pickled results produced by main.py')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Output directory for regenerated plots')
    args = parser.parse_args()

    infile = args.infile
    outdir = args.outdir
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Saved results not found: {infile}")

    with open(infile, 'rb') as fh:
        results = pickle.load(fh)

    # infer max_iter from stored curves (take first algorithm)
    some_algo = next(iter(results.keys()))
    curves = results[some_algo]['curves']
    if hasattr(curves, 'shape') and curves.ndim == 2:
        max_iter = curves.shape[1]
    else:
        # fallback to 300 if shape not found
        max_iter = 300

    print(f"Loaded results from {infile}. max_iter={max_iter}")

    # regenerate the two plots without titles
    main.plot_convergence(results, max_iter, outdir, show_title=False)
    main.plot_violin(results, outdir, show_title=False)

    print("Re-generated convergence and violin plots (titles removed).")


if __name__ == '__main__':
    main_cmd()
