# Blob-Net SI

Standalone LaTeX companion matching the main manuscript typography and authors.

The files in this directory are the curated, submission-ready SI. Only figures
referenced by `supplementary_information.tex` are tracked.

From the BlobNet repository root:

```sh
uv run blobnet-reproduce-publication --target si --device auto --compile-latex
```

Use `uv run blobnet-supplemental-figures --figure gold` to regenerate only the
gold-in-TiO2 figure. Numbered experiment targets and `--figure all` write their
working products under `outputs/`; caches and training intermediates are not
part of this tracked publication package.

Build the document with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_information.tex
```

Figure source data and regeneration caches are written under `outputs/` and
are not tracked with this publication package. Generation code remains in
`scripts/make_supplemental_figures.py` rather than being duplicated here.
