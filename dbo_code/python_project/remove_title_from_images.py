"""
Crop the top area of figure PNGs to remove titles and optionally save as PDF.

Default behavior: process `results/convergence.png` and `results/violin.png`,
remove `crop_top` pixels from the top and write `*_notitle.png` and `*_notitle.pdf`.

Usage examples:
  python remove_title_from_images.py --indir results --files convergence.png violin.png
  python remove_title_from_images.py --indir results --files convergence.png --crop-top 80

Options:
  --crop-top INT     Pixels to remove from the top of the image (default: 70)
  --crop-percent F   Alternatively, fraction of image height to remove (0.0-0.5)
  --overwrite        Overwrite original files (USE WITH CAUTION)
  --outfile-suffix   Suffix to append to output files (default: _notitle)

Requirements:
  Pillow (PIL) - install with: pip install Pillow
"""
import os
import argparse
from PIL import Image


def process_image(path, crop_top=None, crop_percent=None, out_suffix='_notitle', overwrite=False):
    img = Image.open(path)
    w, h = img.size
    if crop_percent is not None:
        if not (0.0 <= crop_percent < 0.5):
            raise ValueError('crop_percent must be between 0.0 and 0.5')
        crop_top_px = int(h * crop_percent)
    else:
        crop_top_px = int(crop_top or 70)
    if crop_top_px <= 0:
        print(f'  [skip] crop_top_px <= 0 for {path}')
        return
    if crop_top_px >= h-10:
        raise ValueError('crop_top too large for image height')

    box = (0, crop_top_px, w, h)
    cropped = img.crop(box)

    base, ext = os.path.splitext(path)
    out_png = f'{base}{out_suffix}.png'
    out_pdf = f'{base}{out_suffix}.pdf'

    if overwrite:
        out_png = path

    cropped.save(out_png)
    # save also as pdf for neat inclusion in papers
    try:
        rgb = cropped.convert('RGB')
        rgb.save(out_pdf)
    except Exception as e:
        print(f'  [warn] could not save PDF for {path}: {e}')

    print(f'  [saved] {out_png}')
    print(f'  [saved] {out_pdf}')


def main():
    parser = argparse.ArgumentParser(description='Remove title area from PNG figure(s)')
    parser.add_argument('--indir', type=str, default='results', help='Directory containing images')
    parser.add_argument('--files', nargs='+', default=['convergence.png', 'violin.png'],
                        help='List of PNG filenames to process (relative to indir)')
    parser.add_argument('--crop-top', type=int, default=70, help='Pixels to crop from top')
    parser.add_argument('--crop-percent', type=float, default=None,
                        help='Fraction of image height to crop from top (overrides --crop-top)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite original PNG (dangerous)')
    parser.add_argument('--outfile-suffix', type=str, default='_notitle', help='Suffix for output files')
    args = parser.parse_args()

    for fname in args.files:
        path = os.path.join(args.indir, fname)
        if not os.path.exists(path):
            print(f'  [skip] not found: {path}')
            continue
        try:
            process_image(path, crop_top=args.crop_top, crop_percent=args.crop_percent,
                          out_suffix=args.outfile_suffix, overwrite=args.overwrite)
        except Exception as e:
            print(f'  [error] processing {path}: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
