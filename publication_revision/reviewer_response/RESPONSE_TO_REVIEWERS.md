# Response to Reviewers

We thank the reviewer for identifying substantive problems in the original
submission. We re-audited the implementation, reran the central CartPole study,
and rewrote the paper as a scoped reproducibility audit. The revision no longer
claims a successful learned-CTRL reproduction or cross-domain validation.

## Major comment 1: Original comparison and unusually low CartPole returns

We agree. The revision now directly compares our results with Figure 2(a) of
Lu et al. Because the source reports curves rather than a numerical table, we
inspect the vector paths and report approximate values rounded to the nearest
ten: D3QN about 260 and CTRLg about 310 at 5,000 samples. We explicitly state
that Lu et al. trained to convergence and evaluated ten trials, while their
training-seed count, failure handling, augmentation ratio, state-noise semantics,
and evaluation noise are not reported.

We now define both evaluation settings precisely. Clean evaluation uses our
11-action CartPole simulator with action and state noise disabled. The stress
test uses action and process-state noise with standard deviation 0.05. We no
longer call the latter “CTRL/noisy,” because the original evaluation protocol is
not documented.

The reported-architecture four-layer, 512-unit D3QN attempt remains a failed
reproduction: over five development seeds it obtains 31.19 +/- 6.84 clean
return. A ten-seed process/observation-noise and stop/continue protocol matrix
produces clean means of only 18.35--34.56, so those ambiguities do not explain
the gap. An independently evaluated online-DQN learnability control reaches
431.00 +/- 66.12 clean over ten training seeds, compared with random
27.15 +/- 0.96. Thus the environment is learnable, but our attempted offline
reproduction does not recover the original result.

For the mechanism audit we disclose a different, stabilized learner (two
256-unit layers, no batch normalization, CQL coefficient 0.05). Its real-only
return is 266.09 +/- 162.59 over 30 seeds. This is in the original plot's
numerical range but is not presented as a faithful reproduction.

## Major comment 2: Variability, unequal runs, and missing inference

We agree and removed the old unequal-seed CF-fraction table and all superiority
language based on it. The revised main study uses 30 paired training/data seeds
per arm, a common 100-episode evaluation bank, fixed optimizer updates, fixed
minibatch size, and fixed 50:50 real/synthetic sampling. The training/data seed,
not each evaluation episode, is the inferential unit.

We report seed-level means, standard deviations, medians, paired differences,
95% paired bootstrap intervals, paired t, Wilcoxon, and sign-randomization tests,
plus Holm adjustments across four frozen contrasts. The primary matched result
is oracle CF minus fresh-shared synthetic augmentation on clean evaluation:
-62.78 return points, 95% paired bootstrap interval [-111.31, -14.56]. Oracle
is lower in 20 seeds, higher in 5, and tied in 5. Raw p-values are .0193, .0080,
and .0192; Holm-adjusted values are .0580, .0241, and .0575 for t, Wilcoxon, and
randomization tests. The paper reports this test dependence and does not select
one favorable test.

The interval does not establish that the reduction exceeds our prospectively
specified 25-point practical threshold. Under process noise, the paired mean is
-2.05 and its 90% bootstrap interval lies inside the study-defined +/-5 band;
we describe this only as meeting that operational practical-equivalence rule.

We removed the LunarLander tables because their variation, model validity, and
reward/termination handling do not support an inferential claim.

## Major comment 3: Undefined continuous-action CF, SAC, MuJoCo, and D4RL

We agree. The earlier “aug” MuJoCo condition was not a valid, documented
continuous-action counterfactual intervention, used only two or three seeds,
and could not support the task-sensitivity narrative. The D4RL rows had no CF
condition, and the HalfCheetah CQL result indicated an unvalidated implementation.
All LunarLander, MuJoCo, SAC, D4RL, “aug,” and continuous-action claims and rows
have therefore been removed from the manuscript and appendix.

The retained experiment is discrete and explicit. For each factual transition,
the simulator generates all ten alternative actions under one of three rules:

- fresh-independent synthetic data draws a new noise pair for each action;
- fresh-shared synthetic data draws one independent pair per factual transition
  and reuses it across the ten actions; and
- oracle CF reuses the factual transition's realized action and state noise.

The matched oracle and fresh-shared arms therefore use the same number of noise
draws and the same within-transition sharing structure. They differ in whether
the shared noise is coupled to the factual transition. Every generated arm uses
the same 50:50 replay sampling and 5,000 optimizer updates. We also state the
remaining limitation: this operational contrast combines factual coupling with
real/synthetic redundancy and effective-sample-size effects; it is not a pure
test of identifiability.

The attempted learned BiCoGAN is not used downstream because it fails the
aggregate eligibility rule in all five development seeds: transition fidelity,
terminal consistency, and latent calibration fail in every seed, while action
reconstruction passes in three of five. Its normalized
counterfactual MSE is 21.4--29.5 times the fresh-noise baseline error and its
terminal disagreement is 42.0%--65.9%. These failures preclude a learned-CTRL
claim.

## Minor comment: Bibliographic integrity

We agree that inaccurate or fabricated references are unacceptable. We removed
the embedded bibliography and rebuilt one external bibliography using primary
publisher, proceedings, OpenReview, arXiv, or official project records. Every
active citation key is checked against this file.

The items called out by the reviewer were handled as follows:

- old reference 13 was replaced with the verified AAAI 2024 ACAMDA record;
- old reference 24 was removed because it did not support the monotonic-network
  claim; the revised text cites Runje and Shankaranarayana (ICML 2023) only for
  constrained monotonic networks;
- old reference 25 was removed rather than guessing the reviewer's suggested
  replacement; a separately verified Chen and Du ICML 2025 record now supports
  the retained exogenous-identifiability statement;
- old reference 26 was removed because its title and arXiv identifier were
  mismatched and the work is not needed by the revised argument; and
- old reference 27 was removed because it was mismatched and is not needed by
  the revised argument.

CTRL is now correctly attributed to Lu et al. and identified as the NeurIPS 2020
Workshop on Offline Reinforcement Learning paper/arXiv:2012.09092, rather than a
NeurIPS main-track article. The revised bibliography also verifies BiCoGAN,
CoDA, MoCoDA, ACAMDA, CAIAC, the causal-RL survey, action-sufficient state
representations, AdaRL, CQL, Dueling DQN, Double DQN, monotonic networks,
counterfactual identifiability, and the statistical reference. Unused and
speculative OASIS, BECAUSE, ReFORM, Dreamer, RAD, ZPD, and invariance references
were removed.

## Revised claim and artifacts

The revised claim is deliberately narrow: for this CartPole simulator, factual
dataset, stabilized offline learner, and 50:50 mixing rule, exact reuse of a
factual exogenous-noise realization does not outperform a sibling-sharing-matched
fresh-noise simulator control. This is a simulator-only mechanism result, not a
successful learned-CTRL reproduction and not evidence that counterfactual
augmentation is generally harmful.

The supplementary archive contains the prospectively frozen local manifest,
exact hashed source snapshot, commands, 30 seed-level artifacts, validation and
summary scripts, statistical outputs, tests, and figure-generation code. We use
“prospectively frozen local follow-up,” not “preregistered,” because there was no
external registration. The remote checkout did not expose Git metadata, so the
source hashes and archived files are the provenance record.
The camera-ready manuscript points to the versioned public snapshot at
`https://github.com/ksajan/ECE595-Causal-RL-Study/tree/claramas-2026-revision-v2/publication_revision`.

## Additional publication-compliance changes

The revision remains in the accepted full-paper category and now contains more
than ten non-reference pages without changes to the LNCS style or layout
parameters. Tables, figures, and appendices have been reordered so that each
appears with its introducing text and the bibliography is contiguous at the end.

In accordance with the AAMAS 2026 policy, the Methods section now discloses the
OpenAI Codex tools and model versions used for experiment-audit planning, code
and test scaffolding, statistical and citation cross-checks, and language
editing. It includes the initiating methodology directive and points to the
supplementary verbatim prompt record. It also discloses that OpenAI Deep
Research and Google Deep Research critiques informed revision priorities, while
their verbatim prompts and underlying model identifiers were unavailable in the
supplied records. The disclosure provides clearly labeled non-verbatim scope
reconstructions rather than fabricating metadata. No reported result was taken
from those critiques. The authors retain responsibility for all design
decisions, claims, references, and submitted text.
