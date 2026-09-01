# Continuous-Control Reviewer Validation Plan

## Purpose

Replace the previous underpowered MuJoCo and D4RL pilots with experiments that
answer the reviewer's methodological questions. The old MuJoCo \`aug\` condition
was observation noise, not counterfactual augmentation, and the old D4RL table
contained no augmented condition. Those results are not evidence for the paper.

## Evidence tiers

1. **Correctness pilots:** validate transition semantics, replay insertion,
   simulator compatibility, normalized scores, and checkpoint reload.
2. **Baseline gates:** require SAC and CQL real-only policies to learn plausible
   task behavior before any augmentation comparison is interpreted.
3. **Paired publication runs:** use identical seeds, evaluation banks, update
   budgets, and real-interaction budgets across variants.

Failed gates are reported as failures and stop downstream interpretation.

## Online MuJoCo SAC

Tasks: \`HalfCheetah-v4\`, \`Hopper-v4\`, \`Walker2d-v4\`, and \`Ant-v4\`.

Variants:

- \`real\`: standard SAC replay.
- \`duplicate\`: insert a second copy of each factual transition, controlling for
  replay-buffer occupancy without adding new information.
- \`oracle_cf\`: snapshot the complete MuJoCo physics state, execute the factual
  action, restore the pre-action state, execute a bounded alternative action,
  and restore the factual post-action state before continuing the trajectory.
  The exact alternate transition is inserted into SAC replay.

Continuous interventions use

\`a_cf = clip(a + Normal(0, sigma * action_range), low, high)\`.

The factual and counterfactual variants receive the same number of real
environment interactions and SAC gradient updates. The counterfactual branch
uses one additional simulator query per factual step. Evaluation uses a fixed
clean seed bank and deterministic policy actions.

Pilot:

- one seed per task and variant;
- 20,000 real interactions;
- 20 evaluation episodes.

Publication matrix after the pilot gate:

- 10 paired seeds;
- 1,000,000 real interactions;
- 100 evaluation episodes;
- report mean, standard deviation, median, paired bootstrap intervals, and
  paired per-seed differences.

## Offline D4RL CQL

Datasets: \`halfcheetah-medium-v2\`, \`hopper-medium-v2\`, and
\`walker2d-medium-v2\`.

The HDF5 \`next_observations\` field is used explicitly through a direct-array
replay buffer. \`terminals\` denotes environment termination; \`timeouts\` is not
treated as terminal for Bellman targets.

For each factual transition, restore the recorded \`qpos\` and \`qvel\`, simulate
the factual and an alternative bounded continuous action, and define:

\`u_s = s'_data - s'_sim(s, a)\`

\`u_r = r_data - r_sim(s, a)\`.

Matched variants:

- \`real\`: factual D4RL transitions only.
- \`simulator_mean\`: simulator alternate-action transition without residual.
- \`fresh_residual\`: alternate-action transition plus a residual permuted from
  another factual transition.
- \`factual_residual\`: alternate-action transition plus the factual transition's
  inferred residual.

The last condition is a simulator-residual structural causal model
approximation. It is not called an exact oracle because \`qpos\` and \`qvel\` do not
contain every hidden MuJoCo solver state and the installed simulator differs
from the historical D4RL stack.

All variants use the same CQL batch size, number of updates, evaluation seed
bank, and real/synthetic sampling fraction. Raw undiscounted returns and
official D4RL normalized scores are both reported.

Simulator gate:

- zero pre-action observation reconstruction error;
- factual reward mean absolute error below 0.05;
- factual termination disagreement below 1%;
- report next-observation replay-error quantiles and the fraction below
  thresholds 0.05, 0.10, and 0.50.

Pilot:

- one policy seed per task and variant;
- 50,000 CQL updates;
- 20 evaluation episodes.

Publication matrix after the baseline and simulator gates:

- 10 paired policy seeds;
- 500,000 CQL updates, extended to 1,000,000 if learning curves have not
  stabilized;
- 50 evaluation episodes;
- report raw and normalized paired effects with bootstrap intervals.

### Recorded gate outcomes and configuration correction

Factual replay over 10,000 seeded transitions passed for
`halfcheetah-medium-v2` (next-state NRMSE 0.0363) and `hopper-medium-v2`
(0.0268). `walker2d-medium-v2` failed the predeclared next-state gate (0.1321
versus the 0.10 threshold), despite exact pre-action observation restoration
and low reward error. Walker2d is therefore restricted to the real-only CQL
baseline; simulator-residual variants are excluded rather than interpreted.

The initial 50,000-update diagnostic used d3rlpy defaults and is not a result.
It differed from d3rlpy's official CQL reproduction in the encoder depth,
Lagrange-multiplier schedule, conservative weight, and update budget. The
baseline gate was restarted with three 256-unit hidden layers,
`alpha_learning_rate=0`, `conservative_weight=10`, and 500,000 updates. A fixed
augmentation seed defines one immutable offline dataset, while paired learner
seeds vary initialization and minibatch sampling.

The validated real-only seed 0 artifact is retained as the first publication
baseline because its comparison-defining protocol is identical to the full-queue
real-only configuration. The full real-only queue therefore runs seeds 1--9,
while each synthetic condition runs seeds 0--9. Summaries merge the gate and
full roots, yielding ten paired learner seeds without rerunning an identical
seed-0 baseline.

## Claim boundary

These experiments test whether exact or simulator-residual one-step
counterfactual augmentation helps continuous-control SAC and CQL under explicit
controls. They do not constitute a faithful learned-BiCoGAN CTRL reproduction.
Task-level claims require the full paired seed matrix; pilot results are
diagnostic only.
