# Final Citation-Integrity Check

Audit target: `samplepaper.tex` and `references.bib` in this directory. No
source file was edited. The audit treats citations in comments and prose such
as ``CTRL/noisy'' as non-citations; only active LaTeX citation commands are
counted.

## Summary

| Check | Status | Finding |
|---|---|---|
| Active cite keys are defined | PASS | All 17 active keys in `samplepaper.tex` are defined in `references.bib`. |
| Every bibliography entry is cited | PASS | All 17 bibliography entries are cited at least once; there are no unused entries and no `\\nocite` commands. |
| Citation commands and bibliography file | PASS | The manuscript uses `\\bibliographystyle{splncs04}` and `\\bibliography{references}` consistently. |
| Primary URLs/DOIs resolve | PASS | All 16 URL/DOI targets present in the bibliography returned HTTP 200 during this audit. Pearl has no URL/DOI and is a book entry. |
| Title, author, year, and venue consistency | PASS | The entries agree with the records represented by their supplied primary landing pages/DOIs. The ACCV paper is correctly listed as the 2019 proceedings volume for the 2018 conference. |
| Reviewer’s old reference 13 | PASS | The old numeric entry is not present; the current manuscript cites the verified ACAMDA record as `sun2024acamda`. |
| Reviewer’s old references 24/25/26/27 | PASS | None is active in the current manuscript or bibliography. No unsupported replacement was retained. |
| Suspicious or fabricated active entries | PASS | No active entry is fabricated or mismatched based on the supplied primary records. |

## Key-level cross-check

The active keys and their bibliography entries are identical:

`agarwal2021precipice`, `armengolurpi2024caiac`, `buesing2019cfgps`,
`chen2025exogenous`, `deng2023survey`, `huang2022adarl`, `huang2022asr`,
`jaiswal2019bicogan`, `kumar2020cql`, `lu2020ctrl`, `pearl2009causality`,
`pitis2020coda`, `pitis2022mocoda`, `runje2023monotonic`, `sun2024acamda`,
`vanhasselt2016double`, and `wang2016dueling`.

The following primary targets were checked for reachability: the CTRL workshop
record, Springer BiCoGAN DOI, OpenReview CFGPS record, PMLR Dueling DQN and
CAIAC records, AAAI Double-DQN and ACAMDA DOIs, NeurIPS CQL/CoDA/MoCoDA and
statistical-precipice records, OpenReview AdaRL and survey records, and the
PMLR ASR, monotonic-network, and exogenous-isomorphism records.

## Old reviewer-number mapping

Numeric labels from the earlier manuscript cannot be carried over because the
bibliography was rewritten and the current paper uses BibTeX keys. The safe
mapping is:

| Old label | Current treatment | Action |
|---|---|---|
| 13 | ACAMDA | Use `\\cite{sun2024acamda}` wherever the ACAMDA claim remains. |
| 24 | Unverified/irrelevant in the reviewer report | Keep removed; do not recreate it without a verified primary record and an in-text use. |
| 25 | Reviewer suggested `arXiv:2505.02212` as a possible intended record | Keep removed from this paper; add only after confirming its exact title, authors, and relevance to a retained claim. |
| 26 | Reviewer suggested `arXiv:2407.14653` as a possible intended record | Keep removed from this paper; add only after confirming its exact metadata and relevance. |
| 27 | Could not be identified as a verified publication | Keep removed. Do not cite or include it. |

This is the correct resolution for the current compact audit paper: it no
longer makes the unsupported claims that required those entries, so adding
speculative replacement references would reintroduce the integrity problem.

## Exact fixes

No citation or bibliography fix is required in the two audited files.

Before submission, run BibTeX from the manuscript directory and inspect the
`.blg`/LaTeX log for unresolved references. If the venue requires a DOI for a
specific record, add it only from the same primary page already listed; do not
restore the former numeric bibliography or any unverified entries.

## Scope limitation

This is a bounded integrity audit, not a claim that every statement in the
paper is empirically correct. It verifies citation-key completeness,
entry usage, bibliographic identity, and the reviewer-flagged reference
problem. It does not independently re-review the experimental results.
