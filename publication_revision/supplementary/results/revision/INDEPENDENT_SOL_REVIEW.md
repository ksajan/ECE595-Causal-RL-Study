# Independent Statistical and Causal-RL Review

## Bottom-line verdict

The current manuscript does **not** establish a successful reproduction of CTRL and
does **not** validate learned counterfactual augmentation. The published-size D3QN
failed its development gate, and the monotonic BiCoGAN failed the overall registered
model-quality gate in every one of five development seeds. The existing CartPole fraction,
LunarLander, MuJoCo, and D4RL claims must therefore be removed rather than reconciled
with the new results.

The 30-seed oracle experiment is worth reporting, but only as a **negative,
mechanism-level diagnostic using the true simulator and a stabilized offline
learner**. Its primary result is that reusing the factual transition noise did not
outperform fresh-noise simulator augmentation. It is not a learned-SCM experiment,
not a faithful CTRL reproduction, and not evidence that counterfactual augmentation
generally harms or helps offline RL.

## Evidence and provenance audit

- All 30 expected seed artifacts (1000--1029) are present once, use one manifest
  hash, one source hash, one combined source/manifest hash, one software stack, and
  the frozen non-seed configuration. Every arm contains 100 evaluation episodes.
- The local manifest predates the result files and its SHA-256 matches the artifacts.
  This supports the description **prospectively frozen local protocol**, but not
  **preregistered study**: there is no external time-stamped registration, and Git
  provenance is recorded as `commit: null, dirty: null`.
- Training seed is correctly used as the inferential unit. Evaluation episodes are
  averaged within seed and are not treated as independent replicates.
- Arms are paired within seed by factual dataset, initial network parameters,
  optimizer-update count, minibatch size, evaluation episodes, and simulator seeds.
  The paired analysis is therefore preferable to comparisons of marginal confidence
  intervals.
- The factual datasets contain only 3,398--3,644 transitions (mean 3,503.5), not
  5,000, because trials stop on failure. Approximately 5.2%--6.6% of transitions
  have different terminal status when judged from the noisy state rather than the
  pre-noise state. These protocol choices are material and differ from, or are not
  specified by, Lu et al.
- The clean returns are bounded and strongly heterogeneous, with several seed means
  at the 500 ceiling. Mean-based inference is consequently sensitive to tail and
  ceiling behavior. The negative primary result is nevertheless supported by its
  median, win/loss count, bootstrap interval, paired t-test, and Wilcoxon test.

## Independent recomputation of the 30-seed result

The arm summaries recompute exactly from the raw JSON files:

| Arm | Clean mean +/- SD; median | Process-noise mean +/- SD; median |
|---|---:|---:|
| Random | 26.93 +/- 1.58; 26.75 | 17.08 +/- 1.02; 17.03 |
| Real only | 272.90 +/- 165.71; 201.10 | 28.21 +/- 5.13; 29.31 |
| Fresh-noise augmentation | 335.49 +/- 122.94; 324.98 | 38.61 +/- 2.94; 39.16 |
| Oracle noise reuse | 280.08 +/- 119.55; 227.40 | 36.73 +/- 3.92; 36.93 |

The four frozen contrasts also recompute:

| Frozen contrast | Paired mean delta | 95% paired bootstrap CI | Raw p: t / Wilcoxon / sign-randomization | Holm p: t / Wilcoxon / sign-randomization |
|---|---:|---:|---:|---:|
| Oracle - fresh, clean (primary) | -55.40 | [-98.94, -9.54] | .0245 / .00510 / .0239 | .0734 / .0153 / .0716 |
| Oracle - fresh, process noise | -1.88 | [-2.96, -0.82] | .00206 / .00172 / .00163 | .00824 / .00687 / .00652 |
| Oracle - real, clean | 7.18 | [-49.98, 65.57] | .814 / .705 / .813 | .814 / .705 / .813 |
| Fresh - real, clean | 62.59 | [8.70, 119.19] | .0399 / .0619 / .0398 | .0799 / .1237 / .0796 |

Small changes in bootstrap random seed give essentially the same intervals. For the
primary contrast, oracle is below fresh in 21 seeds, above it in 7, and tied in 2;
the paired median delta is -65.25. This is credible evidence of a negative mean
difference under the frozen configuration.

The primary contrast was singled out in advance, so its raw p-values may be reported
as the primary analysis. Reporting Holm-adjusted values as a familywise sensitivity
analysis is also valid and more conservative. After Holm adjustment across the four
registered contrasts, the primary conclusion is test-dependent: Wilcoxon remains
below .05, whereas the paired t and randomization tests do not. The paper must report
this disagreement and must not select only the favorable test. Holm correction was
applied separately within each test family; it does not control error jointly over
all tests, intervals, and exploratory analyses.

The phrase `negative_but_below_practical_threshold` is misleading for the primary
effect. The point estimate (-55.40) exceeds the 25-point harm threshold in magnitude,
but its interval crosses -25. The valid conclusion is: **the oracle arm had lower
clean mean return, but the data do not establish that the harm exceeds 25 points**.
It is neither a practically meaningful-harm result under the frozen rule nor a
small-harm result.

For process-noise evaluation, the 90% bootstrap interval for oracle minus fresh is
[-2.79, -1.00], wholly inside the frozen equivalence region [-5, 5]. An independent
parametric TOST gives a 90% t interval of [-2.82, -0.94] and maximum one-sided
p = 2.3e-6. Thus it is coherent to say that oracle is statistically lower by about
1.9 points while meeting the predeclared operational criterion for practical
equivalence. This is equivalence of **mean return under this process-noise protocol
and this +/-5 bound**, not equivalence of policies, return distributions, or CTRL
implementations. Because the bound is study-defined rather than externally
validated, prefer “met our prespecified practical-equivalence rule” over
“demonstrated equivalence.”

Oracle minus real on clean evaluation is inconclusive. Fresh minus real is suggestive
at the unadjusted mean level, but is skewed (mean 62.59 versus median 10.50), is not
supported by Wilcoxon, and does not survive Holm adjustment. It cannot support a
confirmatory superiority claim. Oracle-real and fresh-real process-noise contrasts
in `summary.json` were not registered in the manifest; they may be labeled only as
exploratory descriptive checks.

## Hidden design limitations

1. **The oracle contrast does not isolate causal correctness alone.** Oracle CF uses
   one factual noise realization for all ten alternative actions from a transition.
   Fresh augmentation draws ten independent noise realizations. The arms therefore
   differ in factual coupling, within-transition sibling correlation, noise diversity,
   and effective synthetic sample size. A `fresh-shared` control, drawing one new
   noise value per factual transition and reusing it across all alternative actions,
   is needed to separate factual coupling from correlation/diversity.
2. **Both synthetic arms use the true simulator.** Fresh noise is an unusually strong
   population-simulation control, and oracle CF has direct access to unobserved
   simulator noise. Neither is a deployable learned world model. The result tests an
   oracle mechanism, not practical CTRL.
3. **The learned SCM is unusable by the registered criteria.** Across seeds 960--964,
   learned-CF normalized MSE is 21.4--29.5 times fresh-noise MSE, terminal disagreement
   is 42.0%--65.9%, at least one latent standard deviation fails its range in every
   seed, and action reconstruction passes in only three seeds. No downstream learned-CF
   result is admissible. This indicts the present implementation/training recipe, not
   BiCoGAN or CTRL in general.
4. **The “faithful” downstream learner did not reproduce the original scale.** The
   four-by-512 batch-normalized D3QN obtained clean 31.19 +/- 6.84 (Polyak) and
   31.59 +/- 12.08 (hard target) over five development seeds. Process-noise medians
   were 18.75 and 16.24, near random (17.41). The stipulated noisy-return gate of 100
   is not grounded in Lu et al., whose evaluation-noise semantics are unspecified,
   and appears poorly calibrated against the three-seed online sanity result
   (process-noise mean 51.75 +/- 1.45). Failure of that gate is not by itself proof of
   non-reproduction; the very low clean score and protocol mismatch are the stronger
   evidence.
5. **The stabilized learner is a disclosed post-development deviation.** It uses two
   256-unit layers, no batch normalization, CQL alpha 0.05, and 5,000 fixed updates.
   Lu et al. used four 512-unit layers with batch normalization and trained to
   convergence; CQL was not part of CTRL. Similar returns under the stabilized learner
   cannot rescue a reproduction claim.
6. **“Noisy” is not demonstrably Lu et al.'s protocol.** The current primary code
   feeds Gaussian state noise into subsequent physics as process noise. The manuscript
   currently describes observation noise over a hidden clean state. Lu et al. say only
   that 5% Gaussian noise is added to states and actions. Rename the condition
   “our process-noise stress test,” state the ambiguity, and make the manuscript match
   the executed code.
7. **Reproducibility metadata are incomplete.** Hashes are internally consistent, but
   the artifacts lack a Git commit and dirty-state record. Archive the exact hashed
   source and manifest with the revision rather than claiming commit-level provenance.

## Descriptive comparison with Lu et al. Figure 2(a)

Vector-path inspection of Figure 2(a), rounded to the nearest ten, gives approximately
**260 for D3QN and 310 for CTRLg at 5,000 samples**. The protocol's current value of
approximately 280 for D3QN appears to read the wrong curve and should be corrected.
Lu et al. show error bars but do not define them as uncertainty over independent
training seeds; the text states only that policies were evaluated for ten trials with
different random seeds.

A publication-safe comparison table should include:

- Lu et al.: about 260 D3QN and 310 CTRLg at 5,000 transitions, training to
  convergence, ten evaluation trials, and unspecified failure handling, mixing ratio,
  state-noise semantics, and evaluation-noise condition.
- Our published-size development attempt: about 3,500 transitions because failed
  trials stop, fixed 10,000 updates, and clean real-only means near 31 over five seeds.
- Our stabilized oracle study: real 272.90 +/- 165.71, fresh 335.49 +/- 122.94, and
  oracle 280.08 +/- 119.55 over 30 training/data seeds, but only about 3,500 factual
  transitions, a different architecture, CQL, fixed 5,000 updates, explicit clean
  evaluation, and no learned SCM.

These rows are descriptive context only. Do not compute cross-paper p-values, do not
compare overlapping error bars, and do not say that the stabilized oracle study
matches or exceeds CTRL. The only defensible reproduction conclusion is that the
published-size attempt did not recover Lu et al.'s reported performance and that
underspecified original details prevent attribution of the discrepancy.

## Exact publication-safe claims

The revision may make the following claims, with the stated scope:

1. “Our published-size D3QN attempt did not recover the CartPole performance shown by
   Lu et al.; important training, failure-handling, augmentation, and noise details are
   not specified in the original report.”
2. “A separately stabilized offline learner was evaluated in a prospectively frozen
   30-seed oracle study; this learner and study are not a faithful CTRL reproduction.”
3. “Under clean evaluation, oracle reuse of the factual simulator noise yielded a
   paired mean return 55.40 points below fresh-noise simulator augmentation (95%
   paired bootstrap CI [-98.94, -9.54]; 21 negative, 7 positive, and 2 tied seeds).”
4. “The raw primary tests supported a negative difference, while familywise-adjusted
   evidence was test-dependent (Holm p=.073 t-test, .015 Wilcoxon, and .072
   randomization); the magnitude was not shown to exceed the prespecified 25-point
   practical-harm threshold.”
5. “Under our process-noise stress test, oracle was 1.88 points lower than fresh
   augmentation and met the prespecified +/-5-point practical-equivalence rule.”
6. “Oracle augmentation was inconclusive relative to real-only training on clean
   evaluation. Fresh-noise augmentation versus real-only was suggestive but not robust
   to the registered multiplicity and rank-based sensitivity analyses.”
7. “The oracle result shows that exact factual-noise reuse is not sufficient to
   outperform population-level simulator augmentation for this dataset, learner, and
   mixing rule. The design cannot distinguish coupling from synthetic-pool diversity.”
8. “The attempted monotonic BiCoGAN failed all five development-seed model gates, so
   we do not report a learned-counterfactual downstream claim.”
9. “Three online sanity runs show that the implemented environment is learnable, but
   this small descriptive control is neither an offline baseline nor evidence about
   counterfactual augmentation.”

## Claims that must be withdrawn or rewritten

Remove the following claims from the abstract, contributions, results, discussion,
conclusion, captions, and appendix:

- “verified,” “successful,” “faithful,” or “ground-up reproduction of CTRL”; replace
  with “attempted reimplementation and diagnostic audit.”
- Any statement that the learned BiCoGAN captures the SCM, recovers exogenous noise,
  satisfies identifiability, produces physically consistent counterfactuals, or has
  been validated downstream.
- Any claim that learned CF is conditionally beneficial, improves clean CartPole,
  improves robustness, or depends non-monotonically on CF fraction, dataset size, or
  generator quality. The old unequal-seed fraction and ablation tables are exploratory
  development history and cannot support the revised paper.
- The “imagination bottleneck,” “coverage-versus-bias mechanism,” Bellman divergence
  loop, causal invariance, and task-stability explanations as empirical findings. They
  may appear only as clearly labeled hypotheses or future experiments.
- All LunarLander, MuJoCo, SAC, D4RL, and continuous-action augmentation findings.
  Their conditions are underpowered, incompletely defined, or do not contain a valid
  CF comparison; the D4RL CQL baseline is itself invalid.
- Any assertion that Ant/HalfCheetah benefit while Hopper/Walker2d fail because of
  balance or monotonicity. The existing runs cannot identify that mechanism.
- Any direct statistical comparison, equivalence claim, or “matching” claim against
  Lu et al. Figure 2(a).
- “CTRL/noisy” as a known replication of the original evaluation protocol. Use “our
  process-noise stress test.”
- “D3QN+CQL used by the original CTRL paper.” Lu et al. report D3QN; CQL is this
  study's stabilization addition.
- Any universal recommendation that causal or counterfactual augmentation is
  promising, harmful, robust, or a high-variance lever. The supported conclusion is
  narrower: one oracle coupling rule failed to beat a more diverse exact-simulator
  augmentation control, and the present learned model failed validation.

## Recommended paper framing

The strongest honest revision is an **audit and negative-results paper**: (i) document
the failure to recover the published-size D3QN baseline, (ii) report the failed learned
SCM quality gate, and (iii) present the 30-seed oracle experiment as evidence that
instance-specific noise reuse is not automatically beneficial relative to fresh
simulator augmentation. This is scientifically useful because it separates three
questions that the current manuscript conflates: whether the environment/learner is
functional, whether exact counterfactual coupling helps, and whether a learned SCM can
approximate that oracle. Only the first two have usable evidence, and the second has a
negative, configuration-specific answer.
