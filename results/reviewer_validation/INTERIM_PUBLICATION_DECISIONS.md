# Interim Publication Decisions

This file records decisions made while the optional continuous-control matrix
is running. It is not part of the tagged `claramas-2026-revision-v2`
submission.

## Walker2d-v4 SAC: complete 10-seed cohort

Protocol: paired training seeds 0--9, 1,000,000 real environment interactions,
100 deterministic evaluation episodes per seed, and common evaluation seeds.
All variants use the same SAC update budget. `duplicate` inserts a second copy
of each factual transition; `oracle_cf` restores the exact MuJoCo simulator
state and executes a bounded alternative action.

| Variant | Mean return | SD | Paired delta vs. real | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|
| Real only | 4620.99 | 459.21 | -- | -- |
| Duplicate | 4222.67 | 479.05 | -398.32 | [-741.68, -49.80] |
| Oracle CF | 4597.47 | 400.59 | -23.52 | [-416.41, 371.25] |

The oracle-CF contrast is inconclusive and practically close to zero at the
mean. The duplicate contrast has a bootstrap interval below zero, but its raw
paired tests are marginal (`t` p=0.0630, Wilcoxon p=0.0840,
sign-randomization p=0.0684) and all three Holm-adjusted p-values exceed 0.50.
This test dependence does not support a task-level benefit or harm claim.

**Publication decision:** keep Walker2d and the broader continuous-control
matrix out of revision v2. The result does not answer the reviewer's concern
more strongly than withdrawing the former underpowered cross-domain claim.
Reconsider only after every predeclared task and arm reaches ten paired seeds.
