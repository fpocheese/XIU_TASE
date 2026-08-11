# Readability revision (2026-07-23)

## Source and outputs

- Immutable source: `origin_paper/XIU_tase_paper/main.tex`.
- Clean revision: `revise_0723/main_revised.tex` and PDF.
- Highlighted revision: `revise_0723/main_revision_highlight.tex` and PDF.

## Authorized changes

Implement Parts I--IV of `revision_autoresearch/reviewer1_readability/readability_audit.md`:
restructure and compress the equations in Sections III and IV, add physical intuition,
correct the identified mathematical/notation inconsistencies, and preserve the rest of the
original manuscript unless a cross-reference must be updated.

## Acceptance criteria

1. The original source hash remains
   `e44f660a5af7d7093cb116959aa717b4363ff810a911cb285e41ca40b9f5c2de`.
2. Every original displayed equation in Sections III--IV has an explicit keep/merge/move/delete
   disposition reflected in the revision.
3. No duplicate labels or undefined references are introduced.
4. Clean and latexdiff-highlighted PDFs compile successfully.
5. Compilation logs contain no fatal error and no undefined citation/reference warnings.
