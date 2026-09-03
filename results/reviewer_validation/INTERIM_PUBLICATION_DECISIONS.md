# Interim Publication Decisions

This file records decisions for the optional continuous-control matrix. It is
not part of the tagged `claramas-2026-revision-v2` submission. The complete
matrix was frozen at `final_10seed_20260903T040220Z` after every planned row
reached paired seeds 0--9.

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

The direct oracle-CF minus duplicate-replay contrast is +374.80 return with a
95% paired bootstrap interval of [+37.24, +719.80] (7 positive and 3 negative
seed differences). However, its raw tests do not cross 0.05 and the
Holm-adjusted p-values are 0.291, 0.523, and 0.313. This comparison indicates
that oracle CF avoids the degradation seen from duplicate replay; it does not
show that oracle CF improves on real-only training.

**Publication decision:** keep Walker2d and the broader continuous-control
matrix out of revision v2. The result does not answer the reviewer's concern
more strongly than withdrawing the former underpowered cross-domain claim.
Reconsider only after every predeclared task and arm reaches ten paired seeds.

## HalfCheetah-medium-v2 CQL: complete 10-seed cohort

Protocol: paired training seeds 0--9, 500,000 gradient updates, the full
one-million-transition D4RL dataset, and 50 evaluation episodes per seed. The
primary metric is normalized D4RL score.

| Variant | Mean normalized score | SD | Paired delta vs. real | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|
| Real only | 47.008 | 0.184 | -- | -- |
| Factual residual | 47.395 | 0.327 | +0.387 | [+0.171, +0.588] |
| Fresh residual | 47.354 | 0.174 | +0.346 | [+0.254, +0.433] |
| Simulator mean | 47.426 | 0.278 | +0.417 | [+0.224, +0.609] |

All three paired contrasts are detectable after Holm correction, but their
absolute effects are below 0.5 normalized score points. The factual,
fresh-noise, and simulator variants are also too similar to attribute this
small gain specifically to counterfactual noise reuse.

The direct factual-residual minus fresh-residual contrast confirms this: +0.041
normalized score with a 95% paired bootstrap interval of [-0.095, +0.185]
(6 positive and 4 negative seed differences). The corresponding Holm-adjusted
p-values are 0.597, 0.625, and 0.602, providing no evidence that reusing the
factual residual is better than drawing a fresh residual in this task.

**Publication decision:** do not reopen revision v2 for this result. It verifies
that the corrected CQL baseline is functional, but the augmentation effect is
too small and causally non-specific to strengthen the paper's principal claim.

## Hopper-medium-v2 CQL: complete 10-seed cohort

Protocol: paired training seeds 0--9, 500,000 gradient updates, the full
one-million-transition D4RL dataset, and 50 evaluation episodes per seed. The
primary metric is normalized D4RL score.

| Variant | Mean normalized score | SD | Paired delta vs. real | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|
| Real only | 54.255 | 1.774 | -- | -- |
| Factual residual | 55.342 | 3.250 | +1.087 | [-0.511, +2.720] |
| Fresh residual | 50.362 | 2.607 | -3.893 | [-5.764, -2.280] |
| Simulator mean | 54.230 | 2.308 | -0.025 | [-1.624, +1.983] |

Factual-residual and simulator-mean augmentation are indistinguishable from
real-only training. Fresh-residual augmentation is worse than real-only on all
ten paired seeds; its paired delta remains below zero after Holm correction
(`t` p=0.0128, Wilcoxon p=0.0117, sign-randomization p=0.0117).

The direct factual-residual minus fresh-residual contrast is +4.980 normalized
score with a 95% paired bootstrap interval of [+3.458, +6.661]. All ten paired
differences are positive, and the Holm-adjusted p-values are 0.000535, 0.00391,
and 0.00391. This supports a narrow diagnostic conclusion: preserving the
transition-specific residual avoids the degradation caused by injecting fresh
noise. It does not show that factual-residual augmentation improves over
real-only training.

**Publication decision:** retain this result in the optional validation record,
but do not reopen revision v2. It is scientifically useful evidence for the
importance of noise handling, yet it evaluates simulator-derived residual
controls rather than the paper's learned BiCoGAN counterfactual generator.

## Final online SAC matrix

All four MuJoCo tasks use paired seeds 0--9, 1,000,000 real environment
interactions, 100 deterministic evaluation episodes per seed, common
evaluation seeds, and matched SAC update budgets.

| Task | Real | Duplicate | Oracle CF | Oracle CF delta vs. real [95% CI] |
|---|---:|---:|---:|---:|
| HalfCheetah-v4 | 10904.34 | 11176.52 | 10653.30 | -251.04 [-770.21, +218.64] |
| Hopper-v4 | 2969.76 | 3171.01 | 2910.04 | -59.71 [-644.16, +536.84] |
| Walker2d-v4 | 4620.99 | 4222.67 | 4597.47 | -23.52 [-416.41, +371.25] |
| Ant-v4 | 3842.57 | 3777.98 | 4280.06 | +437.49 [-267.37, +1113.85] |

Every oracle-CF versus real interval includes zero, and every Holm-adjusted
paired test equals 1.00. Direct oracle-CF minus duplicate-replay intervals are
negative for HalfCheetah, inconclusive for Hopper and Ant, and positive for
Walker2d. None of those direct contrasts survives the planned Holm-adjusted
test family (`p >= 0.195`). Thus the task-level point estimates cannot support
a general SAC benefit, harm, or task-sensitivity claim.

## Final publication judgment

The complete matrix strengthens the decision to leave revision v2 unchanged.
It demonstrates that the corrected SAC and CQL baselines run at credible
scales, while showing that simulator-derived one-step augmentation is not
uniformly beneficial. The strongest narrow result is Hopper-medium's
factual-residual advantage over fresh residuals; factual residuals still do
not outperform real-only data. HalfCheetah-medium shows only sub-0.5-point
gains that are shared by factual, fresh, and simulator-mean controls.

These findings are valuable follow-up evidence but do not repair the learned
BiCoGAN's failed eligibility gate and do not constitute a CTRL reproduction.
Adding them to the manuscript would reintroduce continuous-control scope
without strengthening the revised CartPole mechanism claim. Keep the frozen
v2 manuscript and reviewer response as the submission, and retain this matrix
as an auditable post-revision validation artifact.
