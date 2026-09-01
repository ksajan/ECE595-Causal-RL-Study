# Independent Review of the Coupling-Control Study

## Verdict

The follow-up addresses the largest causal confound in the initial 30-seed oracle
study. `fresh_shared` and `oracle_cf` have the same factual states, alternative
actions, number of synthetic transitions, one exogenous-noise pair per factual
transition, and exact reuse of that pair across the ten sibling actions. They differ
in the intended factor: `oracle_cf` reuses the noise realized in the factual
transition, whereas `fresh_shared` uses a new draw from the same noise law.

Under this frozen configuration, oracle factual-noise reuse produced lower clean
mean return than fresh shared-noise augmentation. This is a useful negative
mechanism result. It is not evidence that counterfactual reasoning is generally
harmful, that CTRL fails in general, or that a learned Structural Causal Model (SCM)
cannot work.

## Artifact and provenance audit

- All 30 expected artifacts for seeds 1030--1059 are present exactly once.
- The artifacts share the frozen manifest hash
  `0dbb740ba4045badc105be161631e60cc8fa392e6e25d100bcd8accd0458a257`,
  one source digest, one source/manifest digest, and one software configuration.
- Re-running the validator and summarizer from the frozen source reproduces
  `summary.json` exactly.
- The manifest designates `oracle_cf_minus_fresh_shared_clean` as the primary
  contrast before these runs. This supports the phrase **prospectively frozen local
  follow-up**, not **preregistered experiment**, because no external time-stamped
  registration exists.
- Git provenance remains incomplete (`commit: null`, `dirty: null`). Publication
  artifacts should therefore include the exact hashed source and manifest rather
  than claim commit-level provenance.
- Training seed is the inferential unit. Each seed's 100 evaluation episodes are
  nested measurements and are correctly reduced to one seed-level mean.

## Factual and synthetic-pool alignment

The construction regenerates the factual dataset with the same seed and asserts
exact equality against the shared dataset generator for states, actions, rewards,
next states, terminal labels, and trial identifiers. Independent reconstruction
also found exact alignment of all three synthetic pools for:

- the factual state repeated ten times;
- the ten alternative actions in the same order;
- reward 1 for every transition; and
- the repeated factual trial identifier.

Across seeds, each factual pool contains 3,365--3,705 transitions (mean 3,517.1),
and every synthetic pool contains exactly ten times that count. The lower-than-5,000
factual count is caused by stopping trials at failure.

### Does `fresh_shared` match oracle sibling coupling?

Yes, in the intended structural sense:

- `oracle_cf` uses one factual action-noise scalar and one factual four-dimensional
  state-noise vector for all ten alternatives from a factual transition.
- `fresh_shared` draws one new scalar and one new vector for a factual transition and
  reuses them exactly across its ten alternatives.
- Every seed records zero maximum within-transition reuse error for both action and
  state noise.
- Every seed has one unique shared action-noise value and one unique shared
  state-noise vector per factual transition; both diversity fractions equal 1.0.
- `fresh_independent` instead uses ten noise pairs per factual transition.

Thus `fresh_shared` matches oracle noise-draw count, sibling cluster size, and exact
within-cluster common-noise reuse. “Matched sibling correlation” should be described
this way rather than as an estimated correlation coefficient; no such coefficient
was estimated.

The shared draws also have the intended marginal scale. For representative seed
1030, action noise had mean -0.0007 and standard deviation 0.0511, while each state
noise dimension had standard deviation 0.995--1.012 before multiplication by 0.05.
Across all artifacts, fresh-shared and fresh-independent pools had comparable
normalized discrepancies from oracle (mean normalized MSE 0.174 and 0.173,
respectively).

One implementation blemish should be disclosed: `fresh_shared` and
`fresh_independent` start from the same random-number stream but consume it at
different rates. Consequently, exactly one synthetic next state per seed is shared
between those pools, a fraction of only 0.000027--0.000030. This is too small to
explain the result, but distinct random-number offsets would have made the control
cleaner. The artifacts store aggregate reuse diagnostics rather than the complete
noise arrays, so exact noise verification depends on the frozen source and seeds.

## Independent statistical recomputation

The arm-level seed summaries recompute as follows:

| Arm | Clean mean +/- SD; median | Process-noise mean +/- SD; median |
|---|---:|---:|
| Random | 27.03 +/- 1.52; 27.20 | 17.44 +/- 0.99; 17.60 |
| Real only | 266.09 +/- 162.59; 224.48 | 27.51 +/- 4.81; 27.53 |
| Fresh independent | 322.27 +/- 139.54; 292.78 | 37.94 +/- 2.85; 37.98 |
| Fresh shared | 367.15 +/- 140.23; 429.66 | 37.34 +/- 3.49; 38.25 |
| Oracle CF | 304.37 +/- 136.75; 279.50 | 35.29 +/- 3.52; 35.58 |

All four registered paired contrasts also recompute from the raw files:

| Registered contrast | Mean delta | 95% paired bootstrap CI | Raw p: t / Wilcoxon / randomization | Holm p: t / Wilcoxon / randomization |
|---|---:|---:|---:|---:|
| Oracle - fresh shared, clean (primary) | -62.78 | [-111.31, -14.56] | .0193 / .00804 / .0192 | .0580 / .0241 / .0575 |
| Oracle - fresh shared, process noise | -2.05 | [-2.93, -1.13] | .000140 / .000639 / .000270 | .000561 / .00256 / .00108 |
| Fresh shared - fresh independent, clean | 44.88 | [-16.46, 103.68] | .161 / .0775 / .161 | .321 / .155 / .322 |
| Fresh shared - fresh independent, process noise | -0.60 | [-1.49, 0.31] | .213 / .221 / .210 | .321 / .221 / .322 |

For the primary clean contrast, oracle was lower in 20 seeds, higher in 5, and tied
in 5; the paired median difference was -47.63. Independent bootstrap and
randomization recomputation gives materially identical intervals and p-values.

## Raw, Holm-adjusted, and practical inference

Because one primary contrast was specified before the follow-up runs, its raw
paired interval and raw tests may be presented as the primary analysis. Holm
adjustment across all four registered contrasts is an appropriate familywise
sensitivity analysis. The manuscript must not choose whichever test is favorable:
after Holm adjustment, Wilcoxon remains below .05 for the clean primary contrast,
whereas the paired t-test and sign-randomization test do not. Moreover, the manifest
did not designate one of these three tests as the sole decision test. Report the
effect estimate, interval, all raw tests, and the Holm sensitivity instead of a
binary “statistically significant” label.

The stored label `negative_but_below_practical_threshold` is potentially
misleading. The -62.78 point estimate exceeds the 25-point harm threshold in
magnitude, but the 95% interval crosses -25. The correct interpretation is:

> Oracle reuse had lower clean mean return, but the data do not establish that the
> reduction exceeds the prespecified 25-point practical threshold.

The result also does not satisfy practical equivalence: its 90% bootstrap interval
[-103.83, -21.29] is not contained in [-25, 25].

For process-noise evaluation, oracle was lower than fresh shared by 2.05 points, but
the 90% interval [-2.81, -1.28] lies inside the prespecified [-5, 5] region. It is
publication-safe to say that this comparison **met the prespecified practical-
equivalence rule despite a small negative mean difference**. Do not claim that the
policies or return distributions are equivalent, and note that the +/-5 bound is a
study-defined operational threshold rather than an externally validated standard.

Fresh shared versus fresh independent is inconclusive on clean evaluation. On the
process-noise stress test, it meets the same study-defined practical-equivalence
rule. Therefore the data do not establish that sibling sharing itself improves
clean performance, even though its point estimate is positive.

## What the follow-up does and does not identify

The follow-up substantially improves the initial design. Since oracle and fresh
shared have the same sibling cluster structure and the same number of exogenous
draws, their contrast is no longer explained simply by oracle having less synthetic
noise diversity than a ten-draw-per-transition control.

The remaining interpretation is still conditional. Reusing factual noise also
correlates every synthetic sibling with the real transition included in training;
fresh shared noise does not. Because the learner samples individual transitions and
does not model clusters, the measured effect combines factual coupling, redundancy
between real and synthetic samples, and the resulting effective sample size. That
is the operational consequence of the coupling rule here, not a pure test of causal
identifiability.

Other remaining limits are:

1. Both controls use the exact simulator. There is no learned SCM, latent abduction
   error, model bias, or deployable counterfactual generator in this experiment.
2. Results are specific to one stochastic CartPole implementation, a 50/50 real-to-
   synthetic batch mixture, all ten alternative actions, 5,000 updates, and the
   post-development two-layer D3QN+CQL learner.
3. The stabilized learner differs from Lu et al.'s four-layer D3QN, and CQL was not
   part of the original CTRL method.
4. Trials stop at factual failure, producing about 3,500 rather than 5,000 factual
   transitions. Counterfactual terminal outcomes do not alter trial continuation.
5. State noise is fed into subsequent dynamics as process noise. This is not known
   to be the evaluation semantics used by Lu et al.; call it **our process-noise
   stress test**, not **CTRL/noisy**.
6. The same fixed 100-episode evaluation bank is used for every training seed. This
   improves pairing but conditions inference on that bank and does not estimate
   evaluation-bank variability.
7. Returns are bounded and heterogeneous, with ceiling effects on clean evaluation.
   Mean-based results should be accompanied by medians, seed-level plots, and the
   rank-based sensitivity analysis.
8. The learner configuration was selected during prior development. The frozen
   follow-up protects this comparison from within-study tuning, but it is not an
   independent validation of the learner-selection process.

## Relationship to the initial 30-seed study

The studies should be presented in this order:

1. **Initial oracle study as motivating evidence.** It found oracle minus fresh-
   independent clean return of -55.40 points (95% paired bootstrap CI
   [-98.94, -9.54]) but could not separate factual coupling from sibling correlation
   and noise-draw diversity.
2. **Coupling-control follow-up as the main mechanism result.** It was designed after
   that limitation was identified and prospectively frozen before collecting seeds
   1030--1059. Its oracle-versus-fresh-shared contrast is the better-controlled and
   publication-primary result.
3. **Process-noise and shared-versus-independent contrasts as secondary evidence.**
   They qualify effect magnitude and test whether sharing alone accounts for the
   result.

Do not describe the follow-up as a replication of the initial oracle-versus-
independent effect. In the follow-up cohort, the unregistered oracle-minus-fresh-
independent clean difference is -17.90 points (paired t p=.425; Wilcoxon p=.212),
which does not reproduce the initial effect magnitude. The difference between cohort
means is itself inconclusive, so this is heterogeneity rather than proof of a
contradiction. Do not pool the two cohorts without labeling that analysis post hoc.

The broader paper hierarchy should remain: failed published-size reproduction first,
failed learned-model quality gate second, then the oracle studies as simulator-only
mechanism diagnostics. The coupling-control result cannot rescue a CTRL reproduction
claim or a learned-counterfactual claim.

## Publication-safe interpretation

The following wording is supported:

> We first compared exact factual-noise reuse with fresh simulator noise, but that
> design also changed the dependence and number of noise draws among the ten
> counterfactual siblings. We therefore ran a prospectively frozen 30-seed follow-up
> with a fresh-shared control: one newly sampled noise pair was reused across all ten
> alternative actions for each factual state, matching the oracle arm's sibling
> coupling and noise-draw count without using the factual realization. Under clean
> evaluation, oracle reuse yielded a paired mean return 62.78 points below fresh
> shared augmentation (95% paired bootstrap CI [-111.31, -14.56]; 20 negative, 5
> positive, and 5 tied seeds). Raw paired tests supported a negative difference,
> while Holm-adjusted evidence was test-dependent (p=.058 paired t, .024 Wilcoxon,
> and .058 sign randomization). The interval did not establish that the reduction
> exceeded our prespecified 25-point practical threshold. Under our process-noise
> stress test, the mean difference was -2.05 and met the prespecified +/-5-point
> practical-equivalence rule. Fresh-shared versus fresh-independent augmentation was
> inconclusive on clean evaluation.

The conclusion should be:

> For this simulator, dataset, stabilized offline learner, and 50/50 mixing rule,
> exact reuse of a factual exogenous-noise realization was not sufficient to
> outperform a correlation-matched fresh-noise simulator control. The result isolates
> the operational effect of factual coupling more closely than our initial study, but
> it does not evaluate a learned SCM and does not generalize to CTRL as a method.
