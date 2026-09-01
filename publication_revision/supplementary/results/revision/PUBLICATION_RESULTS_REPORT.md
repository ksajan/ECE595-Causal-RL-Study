# Publication Revision Results

## Evidence decision

The revised evidence does not support a successful learned-CTRL reproduction.
It supports a narrower negative result: with a stabilized offline learner and
exact simulator transitions, reusing the factual transition noise did not
outperform matched synthetic augmentation using an independently drawn shared
noise value. The attempted learned BiCoGAN failed its held-out model gates, so
no learned-counterfactual downstream result is admissible.

## Direct context from Lu et al.

Figure 2(a) of Lu et al. reports results only graphically. Vector-path
inspection, rounded to the nearest ten, gives approximately 260 return for
D3QN and 310 for CTRLg at 5,000 samples. The policies were trained to
convergence and evaluated for ten trials; training-seed uncertainty, failure
handling, augmentation ratio, state-noise semantics, and evaluation noise are
not specified. These values are descriptive reference points, not targets for
cross-paper hypothesis tests.

## Reported-architecture reproduction attempt

The reported four-layer, 512-unit, batch-normalized D3QN did not recover the
original scale on our explicit process-noise protocol. Across five development
seeds, real-only clean return was 31.19 +/- 6.84 with Polyak updates and
31.59 +/- 12.08 with hard target copies. Corresponding process-noise returns
were 18.38 +/- 1.42 and 16.33 +/- 1.13; random was 17.32 +/- 0.70. These are
failed development gates, not confirmatory estimates.

A ten-seed 2x2 protocol diagnostic tested process versus observation noise and
stopping at failure versus continuing every trial to exactly 20 transitions:

| State-noise interpretation | Trial rule | Clean mean +/- SD | Noisy mean +/- SD | Mean factual transitions |
|---|---|---:|---:|---:|
| Process | Stop | 32.34 +/- 18.16 | 18.83 +/- 2.41 | 3,494.3 |
| Process | Continue | 34.56 +/- 11.07 | 19.13 +/- 1.84 | 5,000.0 |
| Observation | Stop | 18.35 +/- 5.72 | 17.97 +/- 5.33 | 4,663.1 |
| Observation | Continue | 21.42 +/- 4.08 | 20.57 +/- 3.34 | 5,000.0 |

Continuing to 5,000 transitions did not recover Lu et al.'s scale. Under the
continue rule, process versus observation noise changed clean return by 13.14
points (95% t interval [6.62, 19.66]), showing that semantics matter but do not
explain the full reproduction gap.

## Environment learnability control

An online DQN selected on a validation seed bank and evaluated on a disjoint
100-episode test bank reached 431.00 +/- 66.12 clean and 46.88 +/- 3.21 under
process noise over ten training seeds. Random-policy values were 27.15 +/- 0.96
and 17.86 +/- 0.52. This establishes that the implemented environment can be
learned; it is not an offline baseline and says nothing by itself about
counterfactual augmentation.

## Initial 30-seed oracle study

The stabilized learner uses two 256-unit hidden layers, no batch normalization,
CQL coefficient 0.05, 5,000 updates, and fixed 50:50 real/synthetic minibatches.
This is a disclosed stabilization extension, not the learner reported by Lu et
al.

| Arm | Clean mean +/- SD | Process-noise mean +/- SD |
|---|---:|---:|
| Random | 26.93 +/- 1.58 | 17.08 +/- 1.02 |
| Real only | 272.90 +/- 165.71 | 28.21 +/- 5.13 |
| Fresh-independent synthetic | 335.49 +/- 122.94 | 38.61 +/- 2.94 |
| Oracle CF | 280.08 +/- 119.55 | 36.73 +/- 3.92 |

Oracle minus fresh-independent clean return was -55.40 (95% paired bootstrap
CI [-98.94, -9.54]); oracle was lower in 21 seeds, higher in 7, and tied in 2.
This study mixed factual coupling with within-transition noise diversity, so it
motivated the matched follow-up below.

## Matched 30-seed coupling-control follow-up

Fresh-shared synthetic augmentation draws one independent action-noise scalar
and one independent four-dimensional state-noise vector per factual transition
and reuses that pair over the same ten alternative actions. It therefore matches
the oracle arm's exact sibling-sharing structure and number of noise draws while
differing in whether the noise came from the factual transition. Factual pools were
asserted exactly equal to the shared reference generator in every run.

| Arm | Clean mean +/- SD; median | Process-noise mean +/- SD; median |
|---|---:|---:|
| Random | 27.03 +/- 1.52; 27.20 | 17.44 +/- 0.99; 17.60 |
| Real only | 266.09 +/- 162.59; 224.48 | 27.51 +/- 4.81; 27.53 |
| Fresh-independent synthetic | 322.27 +/- 139.54; 292.78 | 37.94 +/- 2.85; 37.98 |
| Fresh-shared synthetic | 367.15 +/- 140.23; 429.66 | 37.34 +/- 3.49; 38.25 |
| Oracle CF | 304.37 +/- 136.75; 279.50 | 35.29 +/- 3.52; 35.58 |

Manifest-specified paired contrasts:

| Contrast | Mean delta | 95% paired bootstrap CI | Sign-randomization p / Holm p | Decision |
|---|---:|---:|---:|---|
| Oracle - fresh-shared, clean (primary) | -62.78 | [-111.31, -14.56] | .0192 / .0575 | Lower mean; magnitude not established beyond -25 |
| Oracle - fresh-shared, process noise | -2.05 | [-2.93, -1.13] | .0003 / .0011 | Met study-defined +/-5 practical-equivalence rule |
| Fresh-shared - fresh-independent, clean | 44.88 | [-16.46, 103.68] | .1608 / .3215 | Inconclusive |
| Fresh-shared - fresh-independent, process noise | -0.60 | [-1.49, 0.31] | .2104 / .3215 | Met study-defined +/-5 practical-equivalence rule |

For the primary comparison, oracle was lower in 20 seeds, higher in 5, and tied
in 5; the paired median was -47.63. Raw t, Wilcoxon, and sign-randomization
p-values were .0193, .0080, and .0192. After Holm adjustment across four
manifest-specified contrasts, Wilcoxon remained below .05 (.0241), while t and
randomization did not (.0580 and .0575). The manuscript must report this
test-dependence and the paired interval rather than selecting one favorable
test.

The follow-up should not be described as a replication of the initial
oracle-versus-fresh-independent contrast. Within the follow-up cohort, the
unregistered oracle-minus-fresh-independent clean difference was -17.90 points
and was inconclusive (paired t p=.425; Wilcoxon p=.212). The two cohorts are not
pooled.

One negligible implementation overlap is disclosed for completeness:
fresh-shared and fresh-independent started from the same random-number stream
but consumed it at different rates, leaving exactly one identical synthetic
next state per seed (roughly 0.003% of either pool). This is too small to explain
the observed contrasts, although distinct stream offsets would be preferable in
a future rerun.

## Learned-SCM gate

The positive-weight monotonic BiCoGAN candidate failed the aggregate eligibility
rule in all five development seeds. On an independent 50-trial validation bank:

- learned-CF normalized MSE was 21.4--29.5 times the fresh-noise-to-oracle MSE;
- terminal disagreement was 42.0%--65.9% against a 5% gate;
- every seed had at least one latent dimension outside the [0.5, 2.0] standard
  deviation range; and
- action reconstruction passed its baseline-relative gate in only three of
  five seeds, while each of the other three gates failed in every seed.

These failures show that this implementation/training recipe is not a valid
learned counterfactual generator. They do not establish that BiCoGAN or CTRL is
invalid in general.

## Publication-safe conclusion

The reported-architecture reproduction attempt failed, and protocol ambiguity alone
did not repair it. A stabilized learner recovered clean returns in the same
numerical range as the original plot, but it is not a faithful reproduction.
In two separate 30-seed oracle studies, factual-noise reuse did not beat
fresh simulator augmentation. The matched follow-up shows that this result is
not explained solely by sibling-noise diversity. The narrow supported claim is
that transition-specific noise reuse is not sufficient to outperform the
fresh-shared simulator control under this CartPole dataset, stabilized learner,
and mixing rule. Practical learned
CTRL remains unvalidated because the learned SCM failed held-out gates.

The matched comparison remains operational rather than a pure identifiability
test: factual-noise reuse couples synthetic siblings to the real transition in
the training buffer, so its effect includes real/synthetic redundancy and
effective-sample-size changes.

All LunarLander, MuJoCo, SAC, D4RL, CF-fraction, dataset-size, generator-quality,
and Bellman-selection claims from the earlier manuscript are excluded from the
revised evidence because their protocols or sample sizes do not support them.
