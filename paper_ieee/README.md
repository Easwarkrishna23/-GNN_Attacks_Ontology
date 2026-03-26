# IEEE-Style Project Report (LaTeX)

This folder contains an IEEEtran-based report for the "Adversarial Attacks and Defense Mechanisms in GNNs" project.

## Build

1. Generate tables from the latest CSV results:

```bash
python scripts/make_tables.py
```

2. Compile the PDF:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Figures

Figures are referenced directly from `../results/` (no copies are stored here). If you re-run experiments and regenerate images, re-run `scripts/make_tables.py` and rebuild the LaTeX.

