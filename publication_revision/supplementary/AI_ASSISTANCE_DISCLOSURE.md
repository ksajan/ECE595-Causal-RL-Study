# AI-Assistance Disclosure

## Tools and versions

OpenAI Codex was used from August 31 through September 1, 2026. Delegated tasks
used GPT-5.6 Luna at medium reasoning effort and GPT-5.6 Sol at high reasoning
effort; these labels are taken from the run metadata exposed by the delegation
tool. The tools operated on the authors' local source tree and authorized GPU
machines.

The authors also supplied written critiques produced through OpenAI Deep
Research and Google Deep Research. Those critiques were used to identify
literature gaps, protocol ambiguities, statistical weaknesses, and candidate
revision priorities. The supplied records did not include the initiating
prompts, generation dates, or underlying model identifiers. We therefore report
those fields as unavailable rather than inventing them and provide faithful
scope reconstructions below. Authors should replace the reconstructions with
verbatim records if their account history provides the exact metadata.

## Methodology prompt

The initiating user prompt was:

> Can you create create a plan on whatever improvements we can right now. Run
> them using subagents using luna model as its cheaper to run through them adn
> use them when ever low reasoning is required. Then use sol at high effort to
> analysis and plan next move until you can't think of naything and we reached
> at point where we can edit the latex with comments we got before it gets
> published as everything is done its just before publishing last checks this
> came out so want to fix numbers if we cna and reference is a must.

This directive led to a staged audit: faithful learner and protocol diagnostics,
an exact-simulator counterfactual study, a matched sibling-sharing follow-up,
learned-model quality gates, statistical recomputation, reference verification,
and manuscript revision.

## Focused delegated prompts

The final Luna citation-audit prompt was:

> Perform a bounded final citation-integrity audit for samplepaper.tex and
> references.bib. Check every active citation key, identify unused entries,
> verify title, author, year, venue, URL, and DOI consistency using primary
> records, and flag suspicious or mismatched entries, especially the reviewer's
> old references 13, 24, 25, 26, and 27. Do not edit the manuscript or
> bibliography; write a pass/fail audit with exact fixes.

The first Sol statistics/design review was asked to independently audit the
matched 30-seed coupling-control study, recompute the manifest-specified paired
contrasts from raw seed artifacts, inspect coupling and provenance, distinguish
raw from Holm-adjusted inference, and provide publication-safe wording without
editing source or results.

The final Sol proofread prompt was:

> Perform an independent publication-readiness proofread of the current final
> manuscript, bibliography, reviewer response, citation check, raw result
> summaries, publication report, and rendered PDF. Check every numeric claim,
> scope and causal language, reviewer-comment coverage, internal consistency,
> references, and presentation. Lead with must-fix items and distinguish
> optional edits.

## Deep Research critique records

The exact initiating prompts for the two Deep Research sessions were not
retained in the supplied workspace. The following prompts reconstruct their
scope from the complete outputs and are explicitly **not verbatim**:

> **OpenAI Deep Research scope reconstruction:** Critically review the attached
> CTRL reproduction and extension manuscript. Check whether counterfactual
> generation follows structural-causal abduction and noise reuse, compare the
> implementation and CartPole-SD protocol with Lu et al., audit statistical and
> reproducibility claims, identify missing or incorrect primary citations, and
> give a prioritized revision plan with corrected references.

> **Google Deep Research scope reconstruction:** Perform a comparative
> technical validation of the CTRL reproducibility study for an AAMAS/ALA
> workshop submission. Assess its causal and BiCoGAN foundations, CartPole,
> LunarLander, MuJoCo, and D4RL evidence, failure-mode explanations, statistical
> limitations, workshop suitability, and the experiments or manuscript changes
> needed for a defensible submission.

The supplied outputs had the following scope and product identification:

- **OpenAI Deep Research (model/version unavailable):** a critical manuscript
  review covering noise abduction and reuse, CTRL protocol fidelity,
  compute-matched comparisons, uncertainty reporting, reward and termination
  handling, artifact reproducibility, and reference integrity.
- **Google Deep Research (model/version unavailable):** a comparative workshop
  assessment covering the BiCoGAN pipeline, clean and noisy protocols,
  CartPole-SD, LunarLander, MuJoCo and D4RL claims, baseline limitations,
  statistical power, and possible methodological extensions.

These records were treated as reviewer-style critiques, not as experimental
evidence. No numerical result was copied from them. Every retained citation was
checked against a primary record, and every reported number was regenerated
from the archived raw artifacts.

## How outputs were used

AI assistance was used for planning, code and test scaffolding, command
execution, statistical and citation cross-checking, and prose editing. Frozen
manifests and source hashes were used to prevent post-result design drift. The
reported summaries were regenerated from included per-seed JSON artifacts, and
independent review outputs were treated as diagnostics rather than evidence by
themselves.

No generative image system was used. Figures were produced by Matplotlib from
the included numerical artifacts. The human authors are accountable for the
experimental design, interpretation, accuracy, originality, references, and
final submitted manuscript.
