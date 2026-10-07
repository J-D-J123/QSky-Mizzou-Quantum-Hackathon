# Classical approach report

`classical_approach.tex` is the editable LaTeX source;
`classical_approach.pdf` is the compiled report. It describes Models A–D,
preprocessing, noise mixing, validation, metrics, CUDA/CPU execution, and the
requirements for a matched quantum comparison. It includes citations for DDL,
ESC-50, UrbanSound8K, NASA Small UAS, and the Svanström/ITU ARIS datasets referenced
by the teammate protocol. It reports no new training results.

The report identifies the source snapshot it describes. The file fingerprints in
`classical_approach_sources.json` record that snapshot, including uncommitted work.
Concurrent protocol changes may require a later document update.

Compile from this directory with a standard TeX Live installation (or upload the
`.tex` file to Overleaf):

```bash
pdflatex -interaction=nonstopmode -halt-on-error classical_approach.tex
pdflatex -interaction=nonstopmode -halt-on-error classical_approach.tex
```

Two passes resolve references. No external images, bibliography database, model
artifacts, or training runs are required to build the PDF.
