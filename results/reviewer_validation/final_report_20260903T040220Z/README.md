# Final Continuous-Control Validation

This portable report was generated after the optional reviewer-validation
matrix reached paired learner seeds 0--9 in every planned comparison. It is a
post-revision diagnostic and is not evidence used by the frozen
`claramas-2026-revision-v2` manuscript.

## Protocol

- Online MuJoCo SAC: HalfCheetah-v4, Hopper-v4, Walker2d-v4, and Ant-v4;
  1,000,000 real interactions, 100 evaluation episodes, and matched update
  budgets for real, duplicate-replay, and simulator-oracle CF arms.
- Offline D4RL CQL: HalfCheetah-medium-v2 and Hopper-medium-v2; the full
  one-million-transition datasets, 500,000 gradient updates, 50 evaluation
  episodes, and real, simulator-mean, fresh-residual, and factual-residual
  arms.
- Statistics: paired bootstrap intervals and paired t, Wilcoxon, and exact
  sign-randomization tests with Holm correction across planned contrasts.

## Contents

- `publication/continuous_control_results.md`: concise results and caveats.
- `publication/continuous_control_tables.tex`: publication-ready tables.
- `publication/*.png` and `publication/*.pdf`: reportable effect plots.
- `summary/aggregate_results.*`: absolute arm-level results.
- `summary/paired_summary.*`: each augmentation arm versus real data/replay.
- `summary/matched_control_summary.*`: oracle CF versus duplicate replay and
  factual residual versus fresh residual.
- `FINALIZATION_METADATA.json`: source commit and frozen-run metadata.
- `SHA256SUMS.txt`: integrity manifest for this directory.

## Publication Decision

The matrix does not justify changing revision v2. No online SAC oracle-CF arm
shows a multiplicity-robust advantage over real replay. On D4RL Hopper-medium,
factual residuals outperform fresh residuals by 4.98 normalized points, but do
not outperform real-only data. This supports a narrow noise-handling diagnostic,
not a learned-BiCoGAN CTRL or general counterfactual-augmentation claim.

The full local frozen snapshot, including run-level provenance and checksums,
is `results/reviewer_validation/final_10seed_20260903T040220Z`.
