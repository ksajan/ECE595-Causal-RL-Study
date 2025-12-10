# CTRL Repo Consolidation Plan

Goal: spin out a clean, reproducible repo focused on CTRL and baselines
(D3QN, Rainbow, SAC), keeping useful history and simplifying entry points.

## Target layout
- CTRL/ (core data, models, trainer, utilities, notebooks)
- ctrl_algorithms/ (D3QN, Rainbow, SAC modules)
- scripts/ (train.py, infer.py, eval.py)
- results/{exp}/{YYYYMMDD-HHMMSS}/ (config.json, metrics, plots, checkpoints)
- data/ (placeholder + README on regeneration)
- README.md (quickstart and examples)
- PROJECT_PLAN.md, AGENTS.md, pyproject.toml, uv.lock, docs/ (paper/proposal)

## Migration steps
- [x] Create new repo with history from current project using git filter-repo
  (paths: CTRL, ctrl_algorithms, scripts, results, pyproject.toml, uv.lock,
  README.md, PROJECT_PLAN.md, AGENTS.md, data, PDFs).
- [ ] Prune results to representative plots and add results/README.md with
  run metadata.
- [x] Refactor scripts into:
  - train.py (subcommands: dataset, bicogan, d3qn real/cf, rainbow, sac)
  - infer.py (load policy, run episodes, optional recording)
  - eval.py (batch eval over seeds/checkpoints, summary metrics/plots)
- [ ] Emphasize comparisons: document how replacing the base D3QN with
  Rainbow/SAC affects CTRL outcomes; keep plots/tables that highlight
  real vs CF vs alt models for the class analysis.
- [ ] Standardize run outputs to timestamped directories with config dump.
- [ ] Update README.md to reflect new layout and CLI examples.
- [ ] Run ruff check/format and smoke-test `python scripts/train.py --help`
  and a short dry run.

## Repo name
- ece595-causal-rl-deepdive
