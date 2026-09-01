# Publication Revision Experiment Protocol

## Purpose

This protocol addresses the reviewer concerns about low CartPole returns,
counterfactual semantics, unequal seed counts, missing uncertainty, and
unsupported cross-domain claims. Pilot seeds are used only to debug and freeze
the method. They are excluded from confirmatory inference.

## Evidence tiers

1. **Faithful-reproduction gates (development seeds only):** first validate the real-only
   D3QN against a random policy, then evaluate a positive-in-latent BiCoGAN and
   a conditional triangular-flow extension. These runs select and validate
   frozen configurations; they cannot support manuscript inference.
2. **Oracle-coupling study (30 new seeds):** after the faithful learner gate
   failed, separately test the causal coupling mechanism with a stabilized
   offline learner selected using real-data development runs only. This study
   does not contain a learned SCM and cannot establish a faithful CTRL
   reproduction.
3. **Online sanity control:** select a checkpoint on a validation seed bank and
   report performance on a disjoint test bank. This demonstrates environment
   learnability; it is not an offline or counterfactual baseline.

## CartPole-SD protocol

- Data: the primary protocol uses 250 random-action trials capped at 20
  consecutive transitions and stops a trial when the hidden physics state
  terminates. Lu et al. specify 250 trials of 20 steps but do not document what
  happens when CartPole fails inside a trial. Continuing physics after failure
  to force exactly 5,000 transitions is retained only as an exploratory
  interpretation and cannot support the primary claim.
- Action space: the same 11-action interface in training and both evaluations.
- Clean evaluation: action and observation noise are zero.
- Noisy evaluation: action and state Gaussian standard deviations are
  both 0.05. For the primary reproduction, the noisy next state is fed back
  into the next physics step (process/transition noise), preserving a Markov
  state as assumed by the SCM. A hidden-clean-state interpretation is retained
  only as an explicitly labeled observation-noise ablation.
- Evaluation horizon: 500 steps.
- Confirmatory evaluation bank: 100 episodes with seeds 600000--600099. Earlier
  pilot/gate banks (300000, 400000, and 500000) are excluded from confirmation.
- Offline terminal labels for factual, oracle-CF, and fresh-noise transitions
  are derived from each simulator transition's pre-noise next state. Learned-CF
  terminal status is inferred from the generated next observation and its
  disagreement with the oracle is reported as generator error.

## Paired arms

- **Real only:** D3QN trained only on factual transitions.
- **Fresh noise:** simulator augmentation with new exogenous-noise draws. This is
  a population/interventional augmentation control, not an SCM counterfactual.
- **Oracle CF:** exact simulator transitions that reuse the factual action and
  observation noise under an alternative action.
- **Learned CF:** infer a transition-specific latent and reuse it under each
  alternative action.

Every arm uses the same factual dataset, network initialization, batch size,
number of gradient updates, and evaluation episodes within a seed. Augmented
arms sample 50% real and 50% generated transitions per fixed-size batch.

## Frozen learner and model candidates

- D3QN: four 512-unit hidden layers, batch normalization and ReLU, batch size
  256, discount 0.99, Adam learning rate 1e-4, Polyak target coefficient 0.005,
  no CQL, 10000 final-checkpoint updates.
- BiCoGAN widths: generator 200-400-600-600; encoder and discriminator
  600-600-400-200; 2000 generator-pretraining and 5000 adversarial updates.
- The final SCM is trained on all primary transitions. Model and learned-CF
  diagnostics use a separate 50-trial random-policy dataset generated from a
  disjoint seed; validation does not remove data from the reproduction set.
- The positive-in-latent BiCoGAN uses exponentially parameterized positive
  weights on every path from $u$ to the output, while unconstrained context
  projections condition each layer on state and action. Its encoder and joint
  discriminator are trained adversarially using the published layer widths.
- The stabilization extension uses a conditional lower-triangular affine flow,
  fitted by likelihood and inverted analytically. This is not called BiCoGAN or
  an exact CTRL reproduction.

The triangular model is an identifiable architectural extension, not a claim
that the original authors used the same parameterization.

Publication artifacts carry one mandatory experiment-tier label:

- `ctrl_bicogan_reproduction` uses the positive-in-latent BiCoGAN;
- `unconstrained_bicogan_ablation` uses the unconstrained BiCoGAN; and
- `triangular_flow_extension` uses the likelihood-trained triangular flow.

The scripts reject mismatched tier and generator labels. Development-only
baseline diagnostics are stored under a separate artifact schema.

## Downstream-learner gate

Learner selection uses only real-data performance on development seeds; no CF
outcome is inspected. The candidates are the published-size four-layer D3QN
with either Polyak updates (`tau=0.005`) or hard target copies every 1000
updates. Batch normalization may be removed only as a disclosed stabilization
deviation if both published-size candidates fail. A learner passes when its
median noisy return is at least 100, its median paired advantage over random is
at least 50, and it exceeds random by at least 25 points in four of five seeds.
If no candidate passes, no downstream CF confirmation is run.

The published-size learner did not pass this gate. Therefore, it is reported as
a failed faithful-reproduction attempt and is not silently replaced in that
analysis.

## Oracle-coupling extension

The separate oracle study asks whether instance-specific reuse of simulator
noise helps beyond synthetic transitions generated from independent noise. Its
learner is a disclosed stabilization deviation: two 256-unit hidden layers,
ReLU activations without batch normalization, Adam learning rate $10^{-4}$,
Polyak target coefficient 0.005, CQL coefficient 0.05, and 5,000 updates. These
choices were frozen from real-only development runs before the confirmatory
seeds were evaluated.

The study uses seeds 1000--1029 and four paired arms: random, real only, fresh
noise, and oracle CF. All learned arms use the same factual dataset, initial
network parameters, update count, and evaluation seed banks within each seed.
The two augmented arms use fixed 50:50 real/synthetic minibatches, so the number
of optimizer updates and samples per update is matched.

The primary contrast is oracle CF minus fresh-noise augmentation under clean
evaluation. It isolates transition-specific exogenous-noise reuse from generic
simulator augmentation. A 25-return-point gain is the predeclared smallest
practically meaningful clean effect. Secondary contrasts are oracle CF minus
fresh noise under process-noise evaluation, oracle CF minus real-only under
clean evaluation, and fresh noise minus real-only under clean evaluation. The
process-noise practical threshold is 5 return points.

A benefit is called practically meaningful only when the paired 95% bootstrap
interval lies above the corresponding threshold. A positive effect below the
threshold requires the interval to exclude zero. Practical equivalence requires
the paired 90% bootstrap interval to lie entirely within the negative and
positive threshold. All other outcomes are labeled inconclusive.

## Gate criteria

Before confirmation:

- all result files must have one source hash and identical non-seed configs;
- the complete preregistered seed set must be present exactly once;
- no NaN/Inf losses, predictions, or returns;
- model validation must split whole trajectories, not adjacent transitions;
- held-out alternative-action CF error must be reported against the exact
  simulator oracle;
- learned-CF error on the independent validation set must be at least 20%
  below the fresh-noise-to-oracle error, terminal disagreement must be below
  5%, and every inferred-latent dimension must have standard deviation between
  0.5 and 2.0;
- for BiCoGAN, encoder action-reconstruction MSE must be at least 10% below the
  constant central-action baseline;
- the learned arm must not be interpreted causally if oracle CF is no better
  than fresh-noise augmentation or if held-out CF error is unacceptably large;
- any training-budget or architecture change after inspecting gate results
  requires a new excluded gate.

Gate decisions are machine checked before a result can be summarized or
plotted as confirmatory. Invertible-flow reconstruction of its own abducted
latent is not treated as a predictive-quality metric.

## Direct comparison with Lu et al.

Figure 2(a) of Lu et al. is the only source for their CartPole-SD returns. At
5,000 samples, visual digitization to the nearest five return points gives
approximately 280 for D3QN and 310 for CTRLg. These values have no reported
training-seed uncertainty and are presented only as descriptive reference
points. No cross-paper significance test is performed.

The comparison table must state protocol differences: Lu et al. report 250
trials of 20 consecutive steps, convergence-based training, and ten evaluation
trials, but do not specify failure handling, the augmented-data mixing ratio,
or whether their plotted evaluation contains noise. They describe 5% Gaussian
noise on states and actions but do not state whether state noise is process or
sensor noise; the primary protocol uses the Markov process-noise
interpretation. Our protocol stops data collection on pre-noise failure, fixes
the optimization budget, defines
clean and noisy evaluation explicitly, and treats 30 independent training/data
seeds as the inferential units.

## Confirmatory inference

The registered oracle-study contrasts and practical thresholds are specified in
the preceding section and frozen in
`scripts/revision/oracle_confirmatory_manifest.json`.

A learned-CF contrast is added only if its generator passes every registered
model-quality gate on every development seed.

The training seed is the experimental unit. Report mean and standard deviation,
median and interquartile range, seed-level paired deltas, 95% Student-t and
paired bootstrap intervals, paired t-test, Wilcoxon signed-rank test, paired
sign-randomization test, and Holm-adjusted p-values across the four registered
contrasts. Evaluation episodes are not treated as independent training runs.

## Manuscript exclusions

- Previous CF-fraction tables with unequal seed counts are exploratory and will
  not support superiority claims.
- Previous MuJoCo `aug` rows are removed because `aug` was observation noise,
  not continuous-action counterfactual augmentation.
- Previous D4RL rows are removed because they had no CF condition and the CQL
  baseline was not validated.
- LunarLander remains a negative exploratory diagnostic unless a new matched,
  multi-seed experiment is completed.
- No result may be called an exact reproduction, proof of identifiability, or
  universal robustness improvement.
