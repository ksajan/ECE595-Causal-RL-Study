# AAMAS/ALA Revision Package

This directory contains the publication revision in three parts:

- `manuscript/`: LNCS LaTeX source, verified BibTeX database, final figures,
  class/style files, and compiled PDF
- `reviewer_response/`: point-by-point response and independent citation audit
- `supplementary/`: exact experiment source snapshot, raw seed-level artifacts,
  validation tests, statistical summaries, and reproduction instructions

The revised paper reports a scoped CartPole audit. Unsupported LunarLander,
MuJoCo, SAC, and D4RL claims from the previous manuscript are not included.

Build the manuscript from `manuscript/` with:

```bash
pdflatex samplepaper.tex
bibtex samplepaper
pdflatex samplepaper.tex
pdflatex samplepaper.tex
```

See `supplementary/README.md` for artifact verification and rerun commands.

Verify the archive contents from this directory with:

```bash
sha256sum -c SHA256SUMS.txt
```

The public archival tag is `claramas-2026-revision-v1` in
`https://github.com/ksajan/ECE595-Causal-RL-Study`.
