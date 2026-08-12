# Active manuscript figure backup

This directory contains backup copies of every image file referenced by active (uncommented) `\\includegraphics` commands in `new_highlight/main.tex`.

- Source manuscript: `/home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/revise_0723/new_highlight/main.tex`
- Manuscript SHA-256 at backup time: `ce39c8fe7f6dc39d1b0d2574edca8d0af4ab7cf1c908049f1a688d1b49c18163`
- Active image-reference calls: 65
- Distinct resolved source files: 55 (54 distinct backup files because the two legend sources are byte-identical)
- Missing referenced files: 0
- Layout: flat directory; original basenames are retained unless a collision occurs.
- Collision handling: the two source copies of `standalone_duav_legend.png` are byte-identical; one copy is retained under the original filename `standalone_duav_legend.png`.

See `figure_manifest.csv` for the manuscript line number, LaTeX reference, exact resolved source path, backup filename, and SHA-256 of every active reference.
