# Submission notes (drafted 2026-11-03 IST; nothing has been submitted)

Venue facts below were read from the web on 2026-11-03. Anything marked **UNVERIFIED** I could not confirm. Re-check before relying on it.

## Candidate venues

| Venue | Status | Notes |
|---|---|---|
| Insights from Negative Results in NLP (EMNLP-colocated) | 2026 edition already held (Budapest, 29 Oct 2026). 2027 CFP not found. **Next edition UNVERIFIED.** | Best thematic fit: the paper is a pre-registered negative result. Watch insights-workshop.github.io. |
| ICLR 2027 workshops | Workshop proposal deadline (9 Oct 2026) passed. Workshop acceptance notices due 29 Nov 2026. Suggested workshop-paper deadline 1 Feb 2027, notification 26 Feb 2027 (suggested only; each workshop sets its own). | **No specific memory / agents / negative-results workshop confirmed. UNVERIFIED.** Check the ICLR 2027 workshop list after 29 Nov. |
| ACL 2027 workshops (Kyoto, 21-22 Aug 2027) | Suggested direct-submission deadline 25 Apr 2027, acceptance 22 May, camera-ready 12 Jun (suggested; some workshops differ). | **Specific workshop names not yet known. UNVERIFIED.** |
| SIGIR 2027 (San Jose, 19-24 Jul 2027) | Full-paper deadlines 14 / 21 Jan 2027 (full papers, not a fit for this one). **Reproducibility-track and workshop deadlines TBA / UNVERIFIED.** | Retrieval-heavy paper, so SIGIR reproducibility track is a plausible fit if it opens. |
| ECIR 2027 reproducibility track | Passed (abstract 12 Oct, paper 19 Oct 2026). | Not available. |
| NeurIPS 2026 workshops | Not searched. Almost certainly past deadline. **UNVERIFIED.** | |
| arXiv preprint | Always available in principle, see below. | |

## arXiv

- Natural categories: cs.IR (primary) or cs.CL. Both require endorsement for new submitters.
- Policy (arXiv blog post dated 2026-01-21): a new submitter needs BOTH an institutional email AND prior authorship of an accepted arXiv paper in the domain. Otherwise a personal endorsement from an established author in that domain is needed. arXiv staff cannot endorse or waive this.
- Practical consequence: Krishna needs someone (supervisor, faculty, a researcher he can show the work to) who already publishes in cs.IR or cs.CL and is willing to endorse. Do not buy or beg endorsements from strangers; ask a person who has read the paper.
- Alternative with no endorsement: put the PDF and the repo on GitHub + Zenodo (gives a DOI). It is citable and lists on a CV. It is not peer review.

## What would strengthen it (all free)

1. A 7B reader (Qwen2.5-7B-Instruct, 4-bit) on a free Colab/Kaggle GPU for the QA run. The biggest objection to R-04 is the 1.5B reader.
2. A stronger encoder (e.g. a BGE/E5-class model) for both retrieval and the kNN graph. MiniLM is weak, so the graph is built from weak similarities.
3. An entity or co-reference graph in addition to the MiniLM kNN graph. The current graph is the weakest reasonable graph.
4. Multiple shuffle seeds for the shuffled-W control (currently one seed for the graded-metric hint).
5. A held-out re-tune of w and zeta on a LoCoMo split (the paper deliberately did not tune on LoCoMo).
6. An LLM judge with a free open model, in addition to token-F1.
7. A second confirmatory dataset (for example MultiHop-RAG or HotpotQA-style) before claiming anything about multi-hop in general.
8. Independent pre-registration timestamp (OSF is free) for the next run, so the protocol claim is not self-certified.

## Reviewer-risk list (honest)

1. **Negative result on a self-invented method.** Reviewers may see "method that does not work" and no positive contribution. The paper's value is the audit and protocol; it has to be framed that way and not as a method paper.
2. **Small, weak components.** MiniLM-L6 encoder and a 1.5B reader, CPU-only. "Would it work with a 7B reader / a good encoder?" is the first question. The paper cannot answer it.
3. **Self-registered pre-registration.** Commit timestamps in a personal git repo are not a third-party registry. A sceptical reviewer can discount it. It also has a single author, so nobody independent checked the code.
4. **Power and CIs.** N=277 multi-hop questions. The strong config's CI [-0.004, +0.043] still contains the +0.03 margin, so a small real gain is not excluded.
5. **Config transferred from LongMemEval-S.** The S-only config was tuned on LME-S dev. A reviewer could argue it is a strawman for LoCoMo. The paper reports the strong config as well, which partly answers this.
6. **No LLM judge.** Token-F1 under-credits correct paraphrases.
7. **Novelty vs HippoRAG and related graph-RAG work.** The paper is not claiming a better system, but the related-work section is short (seven main references).
8. **Dataset licence.** LoCoMo is CC BY-NC 4.0. Do not redistribute it in the repo. The repo only has result CSVs and code.
9. **Motivation.** The consciousness-inspired origin is mentioned once and disclaimed. Keep it that way. It is a credibility risk.
10. **Affiliation and authorship.** Affiliation is a TODO in main.tex. Single author, student.

## Before posting anywhere

- Confirm affiliation and contact email (TODO in main.tex).
- Decide whether the repo goes public. If so, check no LoCoMo or LongMemEval data files are committed.
- Add the LoCoMo / LongMemEval / Qwen licence statements if a venue asks.
- Rebuild: `.venv/bin/python paper/make_figs.py` then `cd paper && ~/bin/tectonic main.tex`.
