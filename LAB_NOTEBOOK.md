# LAB NOTEBOOK — consciousness-layer

**Append-only, chronological.** Never edit an old entry except to add a dated `ADDENDUM` line below it.
Every number must trace to a file under `results/<run_id>/` (each has `config.json` with git hash,
seed, packages, models, IST start/end). Times are IST (UTC+5:30).

Protocol per run:
1. Write the PRE-REGISTRATION (hypothesis, numeric/directional prediction, falsification criterion,
   config, seed) and **commit it before running** — the commit timestamp is the proof of pre-registration.
2. Run (the script refuses a dirty tree via `conslayer.runlog.Run`).
3. Append RESULTS / OBSERVATIONS / INTERPRETATION / INTUITION UPDATE / DECISION, update
   `EXPERIMENTS.md` (index), `docs/THEORY_LEDGER.md`, and if the design changes, `docs/THEORY_V1.md`
   (new numbered version). Commit.
Dead ends, bugs and aborted runs get entries too.

Run ID format: `R-YYYY-MM-DD-NN`.

---

## R-2026-10-20-01 — v0 synthetic comparison (E1)  [BACKFILLED 2026-10-26, NOT pre-registered]

- **Code:** commit `ea0cdd0` (the v0 run itself predates this notebook; results in `results/v0/`,
  produced by `experiments/run_v0.py`, `analyze_v0.py`, `retrieval_diag_v0.py`, `nodraft_v0.py`).
- **Pre-registration:** none written at the time. The hypotheses H1–H4 in `docs/EXPERIMENT_PLAN.md`
  existed, but no numeric prediction was committed before the run. Treat every v0 result as exploratory.
- **Config (from `results/v0/config.json`):** Qwen2.5-1.5B-Instruct, greedy, bf16, CPU; MiniLM-L6-v2;
  window 4 user turns; k=4; v0 WeightConfig (γ=0.05/turn, arousal emotion, bounded R, global norm,
  symmetric, τ α=0.2 β=0.7 ρ=0.1 θ=0.2, out-degree 8); ζ=0.85, 20 iters, 5 seeds; A* blend_scaled μ=0.7,
  cost 1/(W+ε); λ = median dev C_norm = 0.0532 (dev seed 1, 10 dialogues). Task seed 0, 40 dialogues × 3 probes.
- **Results** (`results/v0/summary.csv`, `per_probe.csv`, `per_probe_nodraft.csv`, `mcnemar.csv`):

| system | correct/120 | 95% Wilson CI |
|---|---|---|
| plain | 0 | 0.000–0.031 |
| self-refine (no memory) | 0 | 0.000–0.031 |
| vector RAG k=4 | 113 | 0.884–0.971 |
| Sandwich λ-trigger (below) | 44 | 0.286–0.456 |
| Sandwich always-refine | 94 | 0.701–0.848 |
| Sandwich shuffled W | 95 | 0.711–0.855 |
| Sandwich retrieval + RAG prompt (no draft) | 114 | (0.895–0.977) |
| full history | 114 | 0.895–0.977 |

- **Observations:** gold fact in the k=4 retrieved set 120/120 for every retriever (`retrieval_recall.csv`)
  ⇒ task saturated for retrieval. The always-refine deficit (94 vs 113, McNemar p<0.001) came from putting the
  wrong Primary draft into the Secondary prompt (removing the draft → 114). Shuffled W = real W (95 vs 94).
  λ trigger: refine rate 51.7%; C_norm highest on update probes (mean 0.142 vs 0.039–0.046) so updates were
  rarely refined → 2/40. AUROC(low C_norm → Sandwich-always wrong) = 0.659, n=120, 26 wrong, no CI (`signal.json`).
- **Interpretation:** on this task the graph adds nothing over vector RAG; the draft-in-prompt and the
  "refine if C low" trigger are harmful design choices. Conflicting facts raise C (audit 18 confirmed).
- **Intuition update:** before: "graph-weighted retrieval + a reflective second pass should beat RAG".
  After: the second pass is only as good as its prompt; draft anchoring is a real failure mode; the
  synthetic task cannot discriminate retrievers, so weights must be tested on a harder/real benchmark.
- **Decision:** move to LongMemEval retrieval (E3, LLM-free) and re-check C_norm as a signal (E2).

---

## 2026-10-26 17:35 IST — infrastructure / environment notes (no run)

- `.venv` was missing on the box at session start (git-ignored; models/ survived). Recreated:
  python 3.13.5, torch CPU wheel, requirements.txt + rank_bm25, scipy, scikit-learn. Existing 30 tests pass.
- Box memory: 16 GB total but only ~3.5 GB *available* at 17:35 IST (other workloads on the shared box).
  This constrains LLM runs (Qwen2.5-1.5B bf16 needs ~3.5 GB). Will re-check before any LLM run.
- Downloaded `longmemeval_s_cleaned.json` from HF `xiaowu0162/longmemeval-cleaned` (sha256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`, 277 MB) into `data/` (git-ignored).
  Facts checked: 500 questions (30 abstention `_abs`), types multi-session 133, temporal 133, knowledge-update 78,
  single-session-user 70, single-session-assistant 56, preference 30; 47.7 sessions and 493.5 turns per question
  (min 396, max 616); 1.79 `has_answer` turns per question on average, 21 questions have none;
  `has_answer` turns: 842 user, 54 assistant; 189,515 unique turn texts.
- Added `src/conslayer/runlog.py` (run directories + dirty-tree refusal), `stats.py` (Wilson, bootstrap,
  sign-flip, McNemar, AUROC+CI, Holm), `theory_v1.py` (vectorised v1 graph), `tests/test_v1.py`. 50/50 tests pass.

---

## R-2026-10-26-01 — LongMemEval-S embedding cache (preprocessing)

**PRE-REGISTRATION (written before running)**
- Purpose: embed all 189,515 unique turn texts + 500 questions with MiniLM-L6-v2 once (normalised, max 256
  word-pieces, longer turns are truncated), so E3 retrievers share identical vectors. No hypothesis.
- Expectation: wall clock 10–40 min on 8 vCPU (unmeasured guess); a sizeable fraction of turns (esp. assistant
  turns) will exceed 256 tokens and be truncated — this is a known limitation that disadvantages dense
  retrieval relative to BM25 (which sees full text).
- Config: `experiments/lme_embed.py --run-id R-2026-10-26-01`, batch 64, CPU. Seed: n/a (deterministic).

ADDENDUM 2026-10-26 17:50 IST (aborted attempt, logged per protocol): first launch of R-2026-10-26-01 aborted
before doing any work — `git commit` failed (no git identity on the box, repo-local identity set to the
previous commit's author "Krishna Sharma <krishna@local>"), so the tree was dirty and `runlog.Run` refused
to start, as designed. No results were produced; `results/R-2026-10-26-01/` was not created. Relaunched after commit.

CORRECTION 2026-10-26 17:43 IST (box clock): the ADDENDUM above is stamped "17:50 IST", but it was written
before commit 3aec799 (17:38:44 +05:30). The stamp was estimated, not read from the clock; the true time was
≈17:37 IST. The "17:35 IST" infra note is likewise approximate (±5 min). From now on every stamp is taken
from `date` at write time.

---

## R-2026-10-26-02 — E3: LongMemEval-S turn retrieval recall@k, v1 graph vs BM25 / cosine (no LLM)

Written 2026-10-26 17:45 IST, before any E3 code was executed on data (the smoke test below runs after this
pre-registration is committed).

**PRE-REGISTRATION**

- Question: does the THEORY_V1 graph (personalised-PageRank activation over the τ-kNN graph with W=S·T·F·R,
  then A* start→goal) retrieve the evidence turns of LongMemEval-S better than plain cosine over the same
  MiniLM embeddings? Which terms matter? Tests ledger V-04, V-05, V-06, V-07 (and P-04..P-08, P-11, P-12 on a
  non-saturated benchmark).
- Data: `xiaowu0162/longmemeval-cleaned`, `longmemeval_s_cleaned.json` (sha256 d6f21ea9…), all 470
  non-abstention questions (all 470 have ≥1 `has_answer` turn). One graph per question over all its haystack
  turns (user + assistant, ≈494 nodes). Node time = days since the first haystack session + turn_index/1440.
  Cue = question embedding only. Embeddings from R-2026-10-26-01.
- Retrievers: bm25, cosine; confirmatory graph family: v1, v1_shuffledW, v1_randomtopo, v1_noS, v1_noT,
  v1_noE, v1_noR, v1_noAstar; exploratory (labelled x_): x_v1_recency (30-day recency on the input),
  x_v1_zeta0.5, x_v1_zeta0.15. v1 defaults from THEORY_V1 §4 except h = 7 days [arbitrary].
  NB on ablations: −S/−T/−E replace the factor by 1 *in W only*; the topology is still built from τ (= β·S
  when mood=0 and R=1), so −S tests S as a weight, not as an edge-selection criterion.
- Metrics: primary = turn_any@10 (any gold turn in the top 10). Secondary: turn_any@1/@5, turn_frac@k,
  sess_any@k, sess_all@k, turn_ndcg@10. 95% percentile bootstrap CIs (B=10,000, seed 0); paired difference
  vs cosine with bootstrap CI and paired sign-flip permutation p (B=20,000); Holm over the 8 confirmatory
  graph retrievers on the primary metric, α = 0.05.
- Predictions (my prior, written down so it can be wrong):
  1. v1 − cosine on turn_any@10: point −0.04, expected range [−0.10, 0.00]. Reason: v1's top 10 = ~5 cosine
     seeds + their graph neighbours, which are similar to the seeds, not to the question. A turn-level
     similarity graph is mostly redundant with cosine.
  2. bm25 vs cosine: no directional prediction; |diff| < 0.10. MiniLM truncation at 256 word-pieces hurts
     long assistant turns.
  3. v1_noR is *identical* to v1 on every question (no learning ⇒ R≡1; THEORY_V1 §4). Any difference = bug.
  4. v1_randomtopo < v1 by ≥ 0.02 (random edges carry activation to irrelevant turns).
     v1_shuffledW within ±0.02 of v1 (seeds dominate the ranking).
  5. |v1_noS − v1|, |v1_noT − v1|, |v1_noE − v1| < 0.02 each. No direction for T (7-day half-life could help
     same-session evidence or hurt multi-session evidence).
  6. start == goal in > 50 % of questions, so |v1_noAstar − v1| < 0.01.
  7. Exploratory: lower ζ moves v1 toward cosine (ζ=0.15 closer to cosine than ζ=0.85). Recency helps
     knowledge-update questions (by_qtype) — no numeric prediction.
- Falsification / decision criteria:
  - V-04 "graph ≥ cosine" is **supported** if the v1 − cosine 95% CI lies above 0 and Holm p < 0.05;
    **refuted** if the CI lies entirely below 0; otherwise "no detectable difference" (with the CI width).
  - V-05 "real W matters" is supported only if both v1_shuffledW and v1_randomtopo are below v1 with CIs
    excluding 0 (ablation_vs_v1.csv). If shuffledW ≈ v1 the weights carry no ranking information here.
  - V-06 (per factor): factor "contributes" if removing it lowers turn_any@10 with CI excluding 0.
  - V-07: A* contributes if v1_noAstar < v1 with CI excluding 0.
  - Decision for E4: run the full synthetic ablation only if some graph term (S, T, E, A*, real W) shows a
    CI-excluding-0 benefit here; otherwise reduced E4 with that reason.
- Config/seeds: everything in `results/R-2026-10-26-02/config.json`; bootstrap seed 0; control_seed 0.
- Compute expectation: ~470 × 11 graph builds (N≈500 dense), 7 processes, 10–40 min.
- Smoke test first: `--limit 10` under run ID R-2026-10-26-02s (only checks the code runs; its numbers are
  not interpreted).

### R-2026-10-26-01 — RESULTS (appended 2026-10-26 17:58 IST)

| item | value |
|---|---|
| status | done |
| start / end | 2026-10-26 17:38:45 / 17:56:38 IST (1,073.5 s wall; 1,063.5 s embedding) |
| git | 3aec799, clean |
| texts embedded | 190,015 (189,515 unique turns + questions) |
| embedding file sha256 | 684ef8b54f26fccf10cbe37aa5796ff9c128800b99502cd4e6ee5cc0e834cfb8 |
| fraction truncated at 256 word-pieces | 0.340 (sample of 3,801 texts, every 50th by length order) |

Observations:
- **Surprise:** 34 % of texts exceed MiniLM's 256-word-piece window, far more than I assumed when I wrote
  "truncation hurts long assistant turns" (I had no number). Dense retrieval sees only the first ~200 words of a
  third of the turns; BM25 sees everything. This biases E3 toward BM25 and must be listed as a limitation of
  every dense/graph result. Possible later fix: chunk long turns, or mean-pool over windows (not done now; it
  would change the node definition).
- Wall time matched the pre-registered 10–40 min expectation (18 min).
- Progress was non-linear because texts were sorted by length (short first).

Interpretation / decision: cache is usable. Proceed to E3 (R-2026-10-26-02) smoke test then full run.

### R-2026-10-26-02s — E3 smoke test (appended 2026-10-26 17:59 IST)
`--limit 10`, commit a5b2039, 9.1 s wall. Code runs end to end, all output files written
(results/R-2026-10-26-02s/). Numbers (n=10) not interpreted, as pre-registered. Runtime ≈0.9 s/question with
7 processes ⇒ full run expected ≈ 7–10 min, below the pre-registered 10–40 min.

---

## R-2026-10-26-03 — E2: does v1 C_norm predict failure? (no LLM)

Written 2026-10-26 17:59 IST, while R-2026-10-26-02 is running and before I have seen any of its output.

**PRE-REGISTRATION**
- Tests ledger P-18 / V-08 ("C < λ ⇒ response insufficient", i.e. low C_norm predicts failure).
- Part A (synthetic, answers from R-2026-10-20-01, no new LLM calls): rebuild each of the 40 dialogues as a
  v1 graph (THEORY_V1 defaults, h = 13.86 turns), compute C_norm for each of the 120 probes with two cues:
  question + logged plain draft (as in v0) and question only. Labels: wrong[sandwich_always] (26 wrong),
  wrong[rag], wrong[sandwich_nodraft]. Score = −C_norm. Baseline scores: −max cosine(cue, turn), −Φ_dyn, and the
  v0-logged C_norm.
- Part B (LongMemEval-S, from R-2026-10-26-02 per_question.csv): label = miss@10 for v1, cosine, bm25
  retrievers; score = −C_norm(v1); baselines −Φ_dyn, −max cosine(question, any turn); and −residual of C_norm
  after linear regression on max cosine (does C_norm add anything beyond retrieval confidence?). Per-qtype
  AUROCs reported as a confound check.
- AUROC with stratified percentile bootstrap 95% CI, B = 5,000, seed 0.
- Predictions:
  1. Part A, wrong[sandwich_always], −C_norm[q+draft]: AUROC ≈ 0.6 (v0 value 0.659 used a different
     normalisation), CI will include 0.5 or barely exclude it (only 26 positives). Main driver expected to be the
     probe-type confound (update probes have high C_norm and are mostly correct for sandwich_always), so I
     expect −C_norm to look predictive for the wrong reason.
  2. Part B, miss@10[v1], −C_norm: AUROC in [0.45, 0.65]; −max cosine will be ≥ −C_norm (retrieval confidence
     is the obvious baseline); −resid(C_norm | maxcos) AUROC CI will include 0.5.
- Falsification: P-18/V-08 counts as **supported** only if an AUROC CI for −C_norm lies above 0.5 in Part B
  *and* the residualised AUROC CI also lies above 0.5 (i.e. it adds information beyond max cosine). If the
  −C_norm CI lies below 0.5, the direction of §5.2 is refuted (high C predicts failure). Otherwise untested /
  "no detectable signal".
- Command: `.venv/bin/python experiments/e2_cnorm_signal.py --run-id R-2026-10-26-03 --e3 results/R-2026-10-26-02`

### R-2026-10-26-02 — RESULTS (appended 2026-10-26 18:01 IST)

Run: commit ef0b518, clean tree; 470 questions; 35.4 s wall (pre-registered
10–40 min — far faster because graphs are per-question, N≈494, dense numpy). Files:
results/R-2026-10-26-02/{per_question,summary,paired_vs_cosine,primary_holm,ablation_vs_v1,by_qtype}.csv, recall_at10.png.

turn_any@k (mean, 95% bootstrap CI), n = 470:

| retriever | @1 | @5 | @10 [CI] | nDCG@10 | sess_all@10 |
|---|---|---|---|---|---|
| bm25 | 0.538 | 0.783 | 0.857 [0.826, 0.887] | 0.639 | 0.840 |
| cosine | 0.353 | 0.749 | 0.855 [0.823, 0.887] | 0.522 | 0.823 |
| v1 | 0.164 | 0.719 | 0.819 [0.783, 0.853] | 0.420 | 0.734 |
| v1_shuffledW | 0.243 | 0.726 | 0.826 [0.791, 0.860] | 0.460 | 0.770 |
| v1_randomtopo | 0.338 | 0.717 | 0.757 [0.719, 0.796] | 0.457 | 0.696 |
| v1_noS | 0.187 | 0.732 | 0.819 [0.783, 0.853] | 0.432 | 0.736 |
| v1_noT | 0.177 | 0.721 | 0.826 [0.789, 0.860] | 0.434 | 0.770 |
| v1_noE | 0.209 | 0.726 | 0.834 [0.800, 0.868] | 0.441 | 0.734 |
| v1_noR | 0.164 | 0.717 | 0.819 [0.783, 0.853] | 0.420 | 0.734 |
| v1_noAstar | 0.164 | 0.713 | 0.817 [0.781, 0.851] | 0.401 | 0.732 |
| x_v1_recency | 0.157 | 0.713 | 0.815 [0.779, 0.849] | 0.403 | 0.670 |
| x_v1_zeta0.5 | 0.272 | 0.749 | 0.836 [0.802, 0.868] | 0.467 | 0.719 |
| x_v1_zeta0.15 | 0.334 | 0.749 | 0.834 [0.800, 0.868] | 0.491 | 0.717 |

Primary (turn_any@10, paired vs cosine, sign-flip p, Holm over 8):
v1 −0.036 [−0.062, −0.011], p = 0.009, Holm 0.055 (wins/losses 11/28); shuffledW −0.030 [−0.055, −0.004],
Holm 0.089; randomtopo −0.098 [−0.128, −0.070], Holm 0.0004; noS −0.036, Holm 0.055; noT −0.030, Holm 0.089;
noE −0.021 [−0.047, +0.004], Holm 0.133; noR −0.036, Holm 0.055; noAstar −0.038 [−0.064, −0.013], Holm 0.030.
bm25 − cosine: @10 +0.002 [−0.034, +0.038]; @1 +0.185 [+0.130, +0.238]; nDCG@10 +0.117 [+0.084, +0.148].

Ablations vs v1 (Δ = variant − v1; turn_any@10 / nDCG@10):
shuffledW +0.006 [−0.015, 0.028] / +0.040 [0.026, 0.055]; randomtopo −0.062 [−0.085, −0.038] / +0.038 [0.020, 0.056];
noS 0.000 / +0.012 [0.004, 0.020]; noT +0.006 [−0.006, 0.019] / +0.014 [0.008, 0.021];
noE +0.015 [0.000, 0.030] / +0.021 [0.011, 0.032]; noR 0.000 / −0.0002; noAstar −0.002 [−0.006, 0] / −0.018 [−0.024, −0.013];
x_recency −0.004 / −0.017 [−0.028, −0.006]; x_ζ0.5 +0.017 [0.000, 0.034] / +0.048; x_ζ0.15 +0.015 [−0.004, 0.034] / +0.071.

By question type (turn_any@10): v1 loses most on multi-session (0.760 vs cosine 0.868); BM25 wins on
temporal-reasoning (0.858 vs 0.748) and loses on single-session-preference (0.433 vs 0.700).
Recency on the input hurt knowledge-update (0.889 vs v1 0.944).

Diagnostics (means): v1 C_norm 0.074, Φ_dyn 0.077, 81 iterations to tol, 2,272 undirected edges per graph
(≈4.6 per node), start ≠ goal in 66.8 % of questions, mean path length 1.87 nodes, goal is a gold turn in 16.4 %
(cosine top-1 is gold in 35.3 %).

Prediction check:
1. v1 − cosine point −0.04, range [−0.10, 0]: **observed −0.036 — correct.**
2. |bm25 − cosine| < 0.10 at @10: correct (+0.002). Not predicted: BM25 is much better at @1 / nDCG.
3. v1_noR identical to v1: **wrong.** @10 and session metrics identical, but 11/470 A* paths differ and
   1 question changes at @5. Cause (verified on one question, ids printed in the session): A* cost −log W is
   not invariant to rescaling W; dropping R halves W_max, which removes a log 2 per-hop penalty and permits
   longer paths. With cost_mode="inverse" the two paths are identical. Not a code bug; an error in THEORY_V1 §4's
   claim, which only holds for P/A. C_norm also doubles (0.149 vs 0.074) because it is normalised by W_max —
   C_norm is not comparable across configs with different W_max.
4. randomtopo < v1 by ≥0.02: correct (−0.062). shuffledW within ±0.02 of v1: correct (+0.006).
5. |Δ| < 0.02 for −S, −T, −E at @10: correct (0.000, +0.006, +0.015), but every removal *raises* nDCG with CI
   excluding 0 — the factors are mildly harmful as weights.
6. start == goal in > 50 %: **wrong** (33 %). |noAstar − v1| < 0.01: correct (−0.002).
7. Lower ζ moves toward cosine: correct (monotone in ζ on @1 and nDCG). Recency helps KU: **wrong** (hurts).

Observations / surprises:
- Spreading activation is the thing that hurts. The ranking gets *worse the more the activation spreads*
  (ζ 0.15 → 0.5 → 0.85: nDCG 0.491 → 0.467 → 0.420; @1 0.334 → 0.272 → 0.164), and none of the graph variants
  beats plain cosine on any metric. The goal node (argmax A) is a hub, not the best match: it is gold only 16 %
  of the time vs 35 % for cosine top-1.
- Shuffling the weights over the same edges *improves* nDCG (+0.040). The real W concentrates activation on
  hubs more than random weights do; W carries no useful ranking signal here.
- Random topology destroys recall@10 but *improves* @1 and nDCG relative to v1, because with random edges the
  activation barely spreads usefully and stays on the seeds (cosine top-5): n_iter 32, C_norm 0.007.
- Multi-session questions (evidence spread over sessions) suffer most: diffusion stays near the top seed's
  neighbourhood, which is the opposite of what a "memory graph that links related events" should do.
- BM25 vs cosine: ties at @10 but BM25 is far better at rank 1. Consistent with 34 % truncation in the
  dense embeddings (R-2026-10-26-01) — a confound, not just "lexical is better".

Interpretation: On a non-saturated benchmark the v1 graph is a worse retriever than the cosine scores it is
built from; the S/T/E weights add nothing at @10 and slightly harm ranking quality; A* adds a little nDCG but
no recall. Pre-registered verdicts: V-04 refuted (CI below 0; Holm-adjusted p = 0.055, so "refuted at the CI
level, borderline after multiplicity"), V-05 refuted for weights (shuffled ≈ real) though topology matters,
V-06 not supported for S, T, E (and mildly refuted on nDCG), V-07 not supported on recall (small nDCG gain).

Intuition update: Before — "the graph is probably redundant with cosine (≈ or slightly worse)". Now — it is
*harmful as a first-stage ranker*, by a mechanism I can name (PPR diffusion towards hubs). The only way the
graph term could add value is as a *re-ranker / expansion on top of a strong first stage* (e.g. add neighbours
of strong hits for multi-hop questions), not as a replacement. That is a different claim and must be tested
on held-out data, not on these numbers.

Decision:
- THEORY_V1 → v1.1: fix §4 (R-rescaling claim; per-hop penalty of −log cost; C_norm comparability), mark ζ=0.85
  and "rank by A" as refuted for retrieval. Propose a v1.2 hybrid ranking (first-stage score + graph bonus)
  and test it as E3b with a dev/test split (R-2026-10-26-04) — parameters chosen on dev only.
- E4 (synthetic full ablation): **reduced**, because no graph term (S, T, E, A*, real W) showed a CI-excluding-0
  benefit on the primary metric (pre-registered rule). Reduced E4 = the E3b hybrid test plus the A*-cost
  ablation already contained in E3.

### R-2026-10-26-03 — RESULTS (appended 2026-10-26 18:02 IST)

Run: commit/dirty/wall = 0fadbba False 32.1 s. Files: results/R-2026-10-26-03/{auroc.csv, partA_per_probe.csv, partB_per_question.csv};
post-hoc: experiments/e2_posthoc_within_ptype.py → posthoc_within_ptype.csv.

AUROC of −score for failure (stratified bootstrap 95% CI, B = 5,000):

| part | label (n, positives) | −C_norm | −max cos | −resid(C_norm \| maxcos) |
|---|---|---|---|---|
| A synthetic | wrong sandwich_always (120, 26) | 0.671 [0.556, 0.782] (q+draft); 0.662 (q only) | 0.601 [0.495, 0.704] | — |
| A synthetic | wrong rag (120, 7) | 0.886 [0.767, 0.971] | 0.645 | — |
| A synthetic | wrong sandwich_nodraft (120, 6) | 0.939 [0.889, 0.978] | 0.624 | — |
| B LongMemEval | miss@10 v1 (470, 85) | 0.653 [0.584, 0.719] | 0.719 [0.658, 0.776] | 0.564 [0.495, 0.630] |
| B LongMemEval | miss@10 cosine (470, 68) | 0.679 [0.601, 0.754] | 0.736 [0.668, 0.800] | 0.590 [0.517, 0.664] |
| B LongMemEval | miss@10 bm25 (470, 67) | 0.625 [0.548, 0.696] | 0.603 [0.526, 0.678] | 0.578 [0.504, 0.648] |

Φ_dyn gives the same AUROCs as C_norm to 3 decimals (it is the trajectory mean of C_norm and the
trajectory converges; Φ_dyn ≈ C_norm(A*)). v0-logged C_norm on synthetic: 0.659 (matches the backfill).

Post-hoc (NOT pre-registered) within probe type, sandwich_always: contradiction (40, 21 wrong) 0.576
[0.393, 0.754]; recall (40, 3) 0.270; update (40, 2) 0.711 [0.395, 0.974]. Mean C_norm by probe type:
contradiction 0.0065, recall 0.0073, update 0.0224; sandwich_always error rate 52.5 % / 7.5 % / 5.0 %.

Prediction check:
1. Part A ≈ 0.6, CI including or barely excluding 0.5: observed 0.671 [0.556, 0.782] — roughly right. The
   predicted confound is real but the mechanism differs slightly: the signal is driven by *contradiction* probes
   (lowest C_norm, 52.5 % wrong); within that type the AUROC is 0.576 with CI including 0.5.
2. Part B −C_norm in [0.45, 0.65]: observed 0.653 for v1 misses — at the top edge; 0.679 for cosine misses —
   above my range. −maxcos ≥ −C_norm: correct for v1 and cosine misses, wrong for bm25 misses (0.603 < 0.625).
   Residual CI includes 0.5: correct for v1 misses (0.564 [0.495, 0.630]); **wrong** for cosine (0.590
   [0.517, 0.664]) and bm25 misses (0.578 [0.504, 0.648]), whose CIs barely exclude 0.5.

Pre-registration flaw (logged so it is not hidden): I listed three Part-B labels but did not name a primary
one, so the falsification rule is ambiguous. Reading it strictly per label: criterion met for cosine and bm25
misses, not for v1 misses. With three labels, a Bonferroni-style 98.3 % interval would likely include 0.5 for
all three (not computed). Verdict: **weak evidence**, not "supported".

Observations:
- The direction of §5.2 (low C ⇒ failure) holds in every pooled analysis; no CI lies below 0.5. This is the
  first paper claim with a consistently positive (if small) signal.
- Most of C_norm's predictive power is retrieval confidence: max cosine alone is better (0.719 vs 0.653), and
  after removing the linear max-cosine component only 0.56–0.59 remains.
- Part A's high AUROCs for rag / no-draft (0.89 / 0.94) rest on 7 and 6 errors, concentrated in contradiction
  probes — too few to trust.
- Φ_dyn adds nothing over C_norm(A*) under the v1 activation rule (it converges); "dynamic Φ" as defined is
  redundant. Would only differ if activation were truncated before convergence.

Interpretation: C_norm is a weak failure detector whose information is mostly "how well does the cue match
anything in memory". Using it as a refine trigger (P-19) is not justified by these numbers: even a perfect
trigger would need AUROC well above 0.7 to beat "always refine" in v0.

Intuition update: Before — "C_norm will look predictive only via confounds". Now — mostly confound
(probe type on synthetic, retrieval confidence on LongMemEval), plus a small residual (~0.06–0.09 AUROC
above chance) that might be real. Worth one confirmatory test on a held-out set with a single pre-named label.

Decision: ledger P-18/V-08 → "weak evidence (R-2026-10-26-03), mostly explained by max-cosine"; Φ_dyn marked
redundant in THEORY_V1 v1.1; keep the trigger default "always refine".

---

## R-2026-10-26-04 — E3b: graph activation as a re-ranking bonus (v1.2 candidate) — PRE-REGISTRATION
Written: 2026-10-26 18:05 IST, before any E3b numbers exist (a 3-question code smoke check ran on the test harness only, no metrics read beyond "runs").
Parent commit at writing: 9ae7ba2 (pre-reg, code and the ledger/v1.1 update are committed together next).

Why: E3 refuted the graph as a *first-stage* retriever, but showed (a) topology matters (random topology −0.062 recall@10 vs v1), and
(b) less diffusion is better. The remaining constructive hypothesis: the graph helps only as a small bonus on top of a strong first stage
(this is how graph-augmented RAG is usually used). This replaces the rest of E4 per the reduction rule logged in R-2026-10-26-02.

Design:
- Data: LongMemEval-S cleaned, same 470 non-abstention questions, same MiniLM cache (sha256 684ef8b5…cfb8). **These questions were already seen in E3**,
  so this is not a fresh test set; the dev/test split below only protects against tuning the hybrid on the test half.
- Split: sha1(question_id) parity: even → dev, odd → test.
- First stages: (1) cosine; (2) RRF(BM25 rank, cosine rank), k = 60.
- Hybrid score = (1−w)·minmax(first) + w·minmax(A), where A = PPR activation with I seeded by the first stage's top-5 (weights ∝ minmax score).
- Grid (12 per first stage): W ∈ {v1 (S·T·F), S-only}, ζ ∈ {0.5, 0.85}, w ∈ {0.1, 0.25, 0.5}. Each also run with shuffled W (control). h = 7 days, everything else is the V1Config default.
- Selection: on dev only, max turn_any@10, ties → nDCG@10, then smaller w. One config per first stage.
- Primary: test-half turn_any@10, chosen hybrid minus its own first stage (paired, bootstrap 95% CI, sign-flip p), Holm over the 2 first stages.
- Secondary: turn_any@5, nDCG@10, sess_all@10, the multi-session subset, and the same config with shuffled W.

Predictions (written before running):
- Primary Δ ≤ +0.01, with a CI that includes 0, for both first stages (i.e. no useful gain).
- The dev-chosen w will be the smallest (0.1), because E3 says diffusion hurts.
- Shuffled W ≈ real W (|Δ| < 0.01), as in E3.

Decision rule: if the primary Δ has CI > 0 after Holm for either first stage AND real W beats shuffled W on test, write THEORY_V1 v1.2 with
the hybrid as the retrieval definition. Otherwise V-12 is refuted, P-12 stays refuted, and the graph's remaining value claim rests only on topology.
CPU estimate: ~1 s/question/7 procs → ~2 min.

### R-2026-10-26-04 — RESULTS   (run 18:05:47–18:06:14 IST, wall 21.7 s compute, commit 65f87ad, clean tree)
Files: results/R-2026-10-26-04/{config.json, per_question.csv, grid_means.csv, test_results.csv, primary_holm.csv}. Split: dev 223, test 247.

Dev selection (turn_any@10):
- cosine first stage: base 0.865, and the best hybrids tie at 0.870 (v1W ζ0.5 w0.1/0.25, v1W ζ0.85 w0.1, Sonly ζ0.5 w0.25). The tie-break picked **v1W, ζ 0.5, w 0.1**.
- RRF first stage: base 0.919, and several hybrids tie at 0.919. The tie-break picked **S-only, ζ 0.5, w 0.1**.

Primary (test, n = 247), hybrid − own first stage, turn_any@10:
- cosine: 0.846 → 0.846, Δ 0.000 [−0.016, +0.016], p 1.0, Holm 1.0.
- RRF: 0.907 → 0.907, Δ 0.000 [0.000, 0.000], p 1.0, Holm 1.0.

Secondary (test):
- nDCG@10: cosine −0.003 [−0.010, +0.003]; RRF −0.004 [−0.014, +0.005].
- Multi-session (n = 66): cosine recall@10 −0.030 [−0.076, 0.000]; RRF nDCG −0.016 [−0.035, −0.002]; RRF sess_all −0.015 [−0.046, 0].
- Shuffled W with the same config gives the same recall@10 as real W. On nDCG, shuffled is slightly *higher* than real (cosine 0.4951 vs 0.4905).
- Larger w hurts on the test half. RRF with v1W, ζ 0.85, w 0.5: 0.874 vs base 0.907.

Prediction check:
- Primary Δ ≤ +0.01 with a CI containing 0: ✓ for both first stages.
- Chosen w = 0.1: ✓ for both.
- Shuffled ≈ real: ✓.

Unplanned but important observation (baseline, not the graph). RRF(BM25, cosine) alone, all 470 questions:
- turn_any@10 = 0.913 [0.887, 0.938].
- vs cosine 0.855: Δ +0.057 [0.028, 0.087], p 0.0002.
- vs BM25 0.857 (E3 file): Δ +0.055 [0.030, 0.081], p < 0.0001.

This is exploratory: the RRF baseline was pre-registered as a first stage, but not as a comparison against BM25 or cosine. It is the single largest gain seen in any run, and it comes from fusing lexical and dense retrieval, not from the graph. It also partly answers the 256-token truncation confound, since BM25 sees the full text.

Interpretation: across 12 configs per first stage, the graph bonus never beat the first stage on held-out questions. A small bonus changes nothing, and a large bonus hurts. Real weights are no better than shuffled ones. V-12 is refuted. There is no evidence for the graph on LongMemEval-S retrieval in any of the forms tested: first stage (E3), weight terms (E3 ablations), or re-ranking bonus (E3b). The only positive graph-side finding is "top-k τ topology beats random topology", and that only says the kNN edges are better than random edges.

Intuition update: the "associative subconscious" (P-12) has nothing to add on a single-hop fact-retrieval benchmark where the question's words or embedding already point at the evidence. If the graph has value, it would be on tasks needing multi-hop chaining (A → B → C where C shares nothing with the query). LongMemEval multi-session is the closest test here, and the graph was *worse* there in every run. This is a real negative, not a power problem: the CIs exclude gains bigger than about +0.016.

Decision: do not write v1.2. Mark V-12 refuted. Recommend RRF(BM25, dense) as the retrieval layer for any further Sandwich work, and keep the graph only as an optional, tested-negative module.

### E5/E6 — SKIPPED (decision logged 2026-10-26 18:07 IST, before running anything)
Reason:
- (1) The v0 synthetic benchmark is saturated for retrieval (recall 120/120 for every retriever). Scaling it up cannot separate graph memory from RAG, so it cannot test the graph's claims; it could only re-measure the prompt effects (V-09, P-19, P-20).
- (2) The discriminating end-to-end test is LongMemEval QA with an LLM reader. That needs either a judge (the official one is GPT-4o, paid, so excluded) or exact-match scoring that is unreliable for free-form answers.
- (3) A CPU 0.5B-class reader over 470 × ~10 retrieved turns is several hours and would mostly measure reader weakness.
- (4) The λ sweep (E6) is moot. E2 showed C_norm adds only ~0.06–0.09 AUROC beyond max-cosine, and the trigger default is already "always refine".

Left for Krishna's decision: a GPU or paid judge for end-to-end QA, and a multi-hop dataset where association could matter.
CPU used in this session: ≈ 18 min embedding + ≈ 2 min of experiments, well under the 8 h budget.

---

## Session 2026-11-03 — infra note (written 2026-11-03 before any run)
- The `.venv/` that the previous session used was **not present** on the box at session start (only `data/cache`, `models/`, and the
  repo). Rebuilt: `python3 -m venv .venv`, CPU torch 2.15.0, transformers 5.22.1, sentence-transformers 6.2.0, numpy 2.5.4, pandas 3.1.0,
  scipy 1.18.1, rank-bm25 0.2.2, nltk 3.10.4. Package versions differ from the 2026-10-26 runs (not recorded then beyond config.json),
  so numeric results of a re-run could differ in the last digits. After the rebuild `pytest` = **51 passed**, same as before.
- Data cache `data/cache/lme_s_minilm.npy` was still there (sha256 not re-checked here; the E3b config recorded it).
- LoCoMo (snap-research/locomo, repo commit 3eb6f2c, 2024-08-13) downloaded to `data/locomo/locomo10.json`,
  sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. **License: CC BY-NC 4.0** (LICENSE.txt in the repo;
  attribution + non-commercial). Academic evaluation is allowed; we do not redistribute the data (`data/` is git-ignored).
  Access: public raw GitHub URL, no login, no cost.
- What I looked at before writing the pre-registrations below (honest disclosure): per-category counts, evidence-list lengths, a few
  example QA items, the list of malformed evidence strings, and the official `task_eval/evaluation.py` F1 code. **No retrieval result
  of any kind on LoCoMo has been seen.**
- Run ID scheme this session: R-2026-11-03-NN.
  - -01 LoCoMo embedding cache + dataset stats (preprocessing)
  - -02 confirmatory RRF(BM25, cosine) on LoCoMo (Run 1 of the brief)
  - -03 LoCoMo multi-hop graph retrieval, pre-registered graph-vs-RRF test (Run 2)
  - -04 end-to-end QA, Qwen2.5-1.5B reader, token-F1 (Run 3; pre-registered after -03 results, design fixed in advance below)

## R-2026-11-03-01 — LoCoMo embedding cache + dataset stats (preprocessing)

**PRE-REGISTRATION (written before running)**
- Purpose: embed every LoCoMo turn and question once with all-MiniLM-L6-v2 (normalised, max 256 word-pieces) so all LoCoMo retrievers share
  vectors; write category/evidence statistics. No hypothesis, no retrieval metrics.
- Fixed preprocessing rules (also in `experiments/locomo_common.py` docstring): node = one turn; text = "`<speaker>: <text>`" plus
  " [image: <blip_caption>]" when the turn has a BLIP caption; node time = days since session_1 + turn_index/1440 (session timestamps parsed
  from e.g. "1:56 pm on 8 May, 2023"); evidence strings cleaned with regex `D:?(\d+):0*(\d+)` → `D<a>:<b>`, ids absent from the conversation
  dropped (9 malformed strings are known: e.g. 'D8:6; D9:17', 'D:11:26', 'D30:05', 'D', and 'D4:36'/'D10:19' that do not exist);
  category names 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial (taken from the official eval code, which treats 1 with
  multi-answer F1, 3 as open-domain with ';' split, 5 as adversarial; content spot-checked; the README does not give the mapping).
- Expectation: 10 conversations, 5,882 turns, 1,986 QA; < 2 min on CPU; < 5 % of turns above 256 word-pieces (LoCoMo turns are short).
- Command: `.venv/bin/python experiments/locomo_embed.py --run-id R-2026-11-03-01`. Seed: n/a.

### R-2026-11-03-01 — results (written after the run)
- Command as pre-registered. 5,882 turns, 7,854 unique texts (turns + questions), 1,986 QA, 13.2 s embedding, **0.0 % texts above 256 word-pieces**
  (so the LongMemEval truncation confound does not exist here). emb sha256 `60dafbda…ba9163`.
- Category table (valid-evidence counts after cleaning): cat 1 multi-hop n=282 (277 with ≥2 evidence turns, mean 3.13 evidence turns, 2 with >10);
  cat 2 temporal n=321 (40 with ≥2); cat 3 open-domain n=96 (50 with ≥2; **4 with 0 valid evidence → excluded**); cat 4 single-hop n=841 (46 with ≥2);
  cat 5 adversarial n=446 (14 with ≥2).
- 4 questions lost one or more evidence ids to cleaning (evidence ids that do not exist in the conversation, or the bare token 'D'). The other
  malformed strings were recovered by the regex (e.g. 'D8:6; D9:17' → 2 ids, 'D:11:26' → D11:26, 'D30:05' → D30:5).
  Consequence: for those few questions the all-evidence target is slightly easier than the authors intended. Impact ≤ 4 of 1,986 questions.
- Surprise: none. Observation: only 277 + 50 + 40 + 46 + 14 questions need ≥ 2 evidence turns, so every per-category multi-evidence CI outside category 1 is wide.
- Decision: proceed to pre-registration of -02/-03/-04. Primary multi-hop population = category 1 with ≥ 2 valid evidence turns (N = 277).

## R-2026-11-03-02 — Confirmatory RRF(BM25, cosine) on LoCoMo

**PRE-REGISTRATION (written and committed before running; no LoCoMo retrieval result has been seen)**
- Why LoCoMo: LongMemEval-S has no clean held-out split. E3/E3b looked at all 470 non-abstention questions and RRF's gain was found there (R-2026-10-26-04, exploratory).
  LoCoMo is a **different dataset** (multi-session two-person chat, short turns, no truncation confound). This is a replication on a different distribution, not a held-out
  split of the same one, and it will be described that way.
- Hypothesis H1: RRF(BM25, cosine; k=60) retrieves the full evidence set more often than cosine alone and than BM25 alone.
- Population: categories 1–4 (multi-hop, temporal, open-domain, single-hop) with ≥ 1 valid evidence turn (expected N = 282 + 321 + 92 + 841 = 1,536); category 5 (adversarial) excluded because its evidence turns
  support a distractor answer, not the question's target.
- **Primary metric: `all_ev@10`** = 1 if every valid evidence turn is in the top-10 turns of the retriever's ranking, else 0 (impossible for the 4 questions with > 10 evidence turns; kept in the denominator).
  **Primary label: the mean of all_ev@10 over the 1,536 questions**; the two primary comparisons are RRF − cosine and RRF − BM25 (paired by question).
- Secondary (reported, not decision-bearing): all_ev@5, any@10, frac@10 (evidence recall), per-category and ≥ 2-evidence subsets, conversation-cluster bootstrap (10 clusters).
- Prediction (numeric): RRF − cosine ∈ [+0.02, +0.08]; RRF − BM25 ∈ [+0.00, +0.06] on all_ev@10. My probability that H1 is confirmed: ~70 %. BM25 on entity-heavy chat may be strong enough to make RRF − BM25 small.
- **Falsifier / decision rule**: H1 confirmed iff both paired differences are > 0 **and** both two-sided paired sign-flip p-values (B = 20,000, seed 0), Holm-adjusted over these 2 comparisons, are < 0.05.
  If either comparison fails, H1 is "not confirmed" and the LME-S gain is reported as not replicating (or replicating for only one baseline).
- CIs: paired percentile bootstrap over questions (B = 10,000, seed 0).
- Config: BM25Okapi defaults on lowercase `[a-z0-9]+` tokens over "`<speaker>: <text>` [+ image caption]", MiniLM cosine on the cached embeddings, RRF k = 60 (`theory_v1.rrf`), ties broken by stable argsort. Seed 0 (no randomness besides the bootstrap).
- Command: `.venv/bin/python experiments/locomo_rrf.py --run-id R-2026-11-03-02`.

## R-2026-11-03-03 — LoCoMo multi-hop evidence retrieval: graph vs RRF (+ controls, ablations)

**PRE-REGISTRATION (written and committed before running -02 or -03)**
- Hypothesis H2: the v1 graph layer (activation spreading seeded from the first stage, rescoring RRF) retrieves the full evidence set of multi-hop questions more often than RRF alone, by ≥ 0.03 absolute.
- Population: **category 1 (multi-hop) with ≥ 2 valid evidence turns, expected N = 277**. Secondary populations: all categories 1–4 with ≥ 2 evidence turns, and each category separately (all and ≥ 2 evidence).
- **Primary metric: `all_ev@10`.** **Primary label: mean all_ev@10 of `rr_sonly` minus mean all_ev@10 of `rrf`, paired over the 277 questions.**
  `rr_sonly` = `V1Graph.hybrid_rank(RRF score, w = 0.1)`, graph with W = S only (T, E off), ζ = 0.5, 8-nearest-neighbour edges with τ > 0.2, seeds = top-5 of RRF. This config is the one **selected on the LongMemEval dev half** (R-2026-10-26-04); it is **not tuned on LoCoMo**.
- Retrievers (each ranks all turns of the conversation; one graph per conversation, ≈ 370–690 nodes, half-life 7 days, T from session timestamps):
  - first stages: `bm25`, `cosine`, `rrf`
  - graph as first stage (v1 `retrieve`: argmax-activation goal, A* path, then activation order): `v1`, controls `v1_shuffledW`, `v1_randomtopo`, ablations `v1_noS`, `v1_noT`, `v1_noE`, `v1_noR`, `v1_noAstar`
  - graph as rerank over RRF: `rr_sonly` (**primary**), controls `rr_sonly_shuf` (seed 0; extra seeds 1, 2 reported for variance), `rr_sonly_rand`; `rr_v1W` (full W, ζ=0.5, w=0.1); `rr_strong` (full W, ζ=0.85, **w = 0.5**, deliberately aggressive so the graph has a fair chance to matter), controls `rr_strong_shuf`, `rr_strong_rand`, ablations `rr_strong_noS/T/E/R`.
  - Note (known before running, from THEORY_V1 §4): R ≡ 1 without learning, so `*_noR` must equal its full model exactly. That is a sanity check, not a finding.
- Prediction (numeric/directional), on cat-1 ≥ 2-evidence all_ev@10:
  - `rr_sonly − rrf` ∈ [−0.01, +0.01] (w = 0.1 rarely changes the top-10 set).
  - `rr_strong − rrf` ∈ [−0.06, +0.02]. `v1 − rrf` ≤ −0.05 (first-stage activation ranking was worse than cosine on LME-S).
  - Real W vs shuffled W within ±0.02 of each other; random topology worse than real topology; each S/T/E ablation within ±0.02 of the full model.
  - My probability that H2 is supported (Δ ≥ 0.03 with the rule below): ~7 %. I expect H2 to be falsified.
- **Falsifier** (both pre-registered): (A) the 95 % CI upper bound of `rr_sonly − rrf` is < 0.03; or (B) `rr_sonly` ≤ `rr_sonly_shuf` on all_ev@10 (point estimate). Either one ⇒ H2 FALSIFIED.
  H2 SUPPORTED only if neither falsifier fires, point Δ ≥ 0.03 and the CI lower bound > 0. Otherwise INCONCLUSIVE.
- Statistics: paired percentile bootstrap (B = 10,000, seed 0) over questions; two-sided paired sign-flip (B = 20,000, seed 0). The primary comparison is a single pre-registered test (α = 0.05, no adjustment). All other comparisons
  (listed in `FAMILY` in `locomo_graph.py`: rrf vs cosine/bm25, v1 vs rrf, rr_strong/rr_v1W vs rrf, real vs shuffled/random for both graphs, 5 `v1` ablations, 4 `rr_strong` ablations; 20 tests) are corrected together with **Holm**, and are labelled exploratory-confirmatory (their direction was predicted but not their size).
  Secondary metrics: all_ev@5, frac@10, any@10. Per-category tables are descriptive.
- Honest caveats fixed in advance: one shuffle seed for the main control (seeds 1–2 for the primary one only); 1 dataset; 10 conversations ⇒ questions are not independent (conversation-cluster sensitivity is reported in -02 only);
  v1 rankings come from `retrieve(k=10)`, so v1-first-stage numbers exist only at k ≤ 10.
- Command: `.venv/bin/python experiments/locomo_graph.py --run-id R-2026-11-03-03`.

## R-2026-11-03-04 — End-to-end QA, Qwen2.5-1.5B reader, LoCoMo token-F1 (design fixed now; run after -02/-03)

**PRE-REGISTRATION (written before -02/-03 results exist; code for this run is committed separately before it is run)**
- Question: does graph context or the Sandwich design change answer quality of a small reader on LoCoMo multi-hop questions?
- Population: category 1, ≥ 2 valid evidence turns (N = 277, all of them; if the run exceeds ~3 h wall-clock the cap is reduced to the first n by sha1(qid) order and **logged**).
- Reader: Qwen2.5-1.5B-Instruct (bf16, CPU, greedy, ≤ 32 new tokens), same model for draft and refine. Metric: LoCoMo official partial-F1 for category 1 (answer split on ',', mean over gold sub-answers of max F1 over predicted sub-answers, Porter stemming, article/punctuation normalisation; reimplemented because the official file imports bert_score/rouge). **No LLM judge, no paid API.**
- Arms (k = 10 turns, shown in chronological order with their session date):
  - `A_rrf`: RRF top-10 context.
  - `B_graph`: `rr_sonly` ranking top-10 (the primary graph arm of -03).
  - `C_sandwich_nodraft`: Sandwich retrieval = v1 first-stage graph ranking top-10, secondary prompt without any draft.
  - `D_sandwich_draft`: as C, plus the "draft" = closed-book answer of the primary pass, shown in the prompt (the original Sandwich design).
  - `Z_closed_book`: no context (this is also D's draft). Sanity arm.
  - Identical prompts are answered once (greedy ⇒ deterministic) and reused; B = A for questions where the top-10 sets in the same order coincide.
- **Primary label: mean F1 of `B_graph` − `A_rrf`** (paired bootstrap). Other comparisons: C − A, D − C, A − Z (Holm over {B−A, C−A, D−C}; A−Z is a sanity check).
- Prediction: A − Z ≥ +0.10 (context helps a 1.5B reader a lot); B − A ∈ [−0.02, +0.02]; C − A < 0 (the v1 first-stage ranking retrieves worse); D − C < 0 (a wrong closed-book draft anchors the refine pass, as in R-2026-10-20-01).
  Absolute F1 for A: 0.15–0.35. Falsifier for "graph helps the reader": CI upper bound of B − A < +0.02.
- Caveat fixed in advance: B and A share most of their context, so B − A has low power by construction; C is the real test of the original retrieval design. 1.5B reader, one seed (greedy), one dataset, partial-F1 rewards lexical overlap.
- Command (script to be added): `.venv/bin/python experiments/locomo_qa.py --run-id R-2026-11-03-04`.

### R-2026-11-03-02 — results (written after the run)
- Command as pre-registered; N = 1,536 (cat 1–4, ≥ 1 valid evidence turn), 10 conversations. Files: `results/R-2026-11-03-02/{summary,paired,primary,cluster_sensitivity}.csv`.
- **Primary (all_ev@10):** cosine 0.391 [0.367, 0.415]; BM25 0.467 [0.443, 0.492]; **RRF 0.493 [0.469, 0.518]**.
  - RRF − cosine = **+0.102** [+0.081, +0.124], sign-flip p_Holm = 1.0e-4.
  - RRF − BM25 = **+0.026** [+0.007, +0.045], sign-flip p_Holm = 0.010.
  - **H1 confirmed** under the pre-registered rule (both diffs > 0, both Holm p < 0.05).
- Prediction check: RRF − BM25 (+0.026) inside the predicted [0.00, +0.06]. **RRF − cosine (+0.102) is outside the predicted [+0.02, +0.08]** — I under-estimated how much worse MiniLM cosine is than lexical match on short, entity-heavy LoCoMo turns.
- Secondary / sensitivity (not decision-bearing):
  - all_ev@5: RRF − BM25 = +0.007 [−0.014, +0.027] (not distinguishable). The RRF gain over BM25 shows at @10, not @5.
  - Conversation-cluster view (mean of 10 per-conversation diffs): RRF − cosine +0.101 [+0.082, +0.121], RRF better in 10/10 conversations; **RRF − BM25 +0.027 [−0.006, +0.063], better in 7/10** — the CI includes 0 once questions are clustered by conversation. So the BM25 comparison is the fragile one.
  - Multi-evidence subset (≥ 2 evidence turns, N = 413): all_ev@10 RRF 0.121, cosine 0.102, BM25 0.070; RRF − cosine = +0.019 [−0.010, +0.048], i.e. on questions that need several turns, RRF is **not** distinguishable from cosine. All three retrievers are poor there (≤ 12 % of questions get all evidence into the top-10).
- Observations: absolute numbers are low for multi-evidence questions; this is the headroom the graph claims to address (see -03).
- Surprises: MiniLM cosine is 7.6 points below BM25 on all_ev@10 (0.391 vs 0.467). I did not predict the cosine-vs-BM25 gap, only the RRF differences.
- Bugs: none in this run. Note: the cluster bootstrap here is a mean of 10 conversation-level diffs with a bootstrap over conversations (n = 10, crude).
- Interpretation: the "hybrid beats either part" finding transfers to a second dataset in direction, with magnitude largest vs the dense retriever. A claim that RRF beats BM25 *robustly* is not supported once conversation clustering is considered. This is a replication on a **different dataset**, not a held-out split of LME-S.
- Intuition update: dense MiniLM-L6 on conversational turns is weak; I should expect any graph built on MiniLM edges to inherit that weakness.
- Decision: proceed to -03 (already pre-registered).

### R-2026-11-03-03 — ABORTED attempt 1 (logged dead end; no result was produced or looked at)
- Attempt 1 was started at 11:31 IST (box clock) with the pre-registered command. After ~10 min wall-clock the pool's 3 slowest workers had used 20+ CPU-minutes each (load average 24 on 8 cores) and no result file existed, while the other workers had finished their conversations in 30–50 CPU-s.
- Diagnosis (py-spy + a single-process timing): `locomo_graph.py` imports numpy **before** `locomo_retr.py` sets `OMP_NUM_THREADS=1`, so 8 worker processes each ran multi-threaded BLAS on tiny (≈600×600) matmuls — oversubscription. With threads pinned to 1, one `v1` retrieval costs 14 ms.
- Action: killed the run, moved its half-written directory to `results/R-2026-11-03-03_ABORTED_attempt1/` (contains only config.json), pinned the BLAS env vars at the top of `locomo_graph.py` (no change to any retriever, metric, config, population or statistic), committed, re-ran under the same run ID. The pre-registration above is unchanged.

### R-2026-11-03-03 — results (written after the run; attempt 2, 77 s wall-clock, commit bf8ddb9, clean tree)
- Command as pre-registered. N = 277 (category 1, ≥ 2 valid evidence turns). Files: `results/R-2026-11-03-03/{per_question,summary,paired_all,family_holm}.csv`, `all_ev_at10_cat1.png`.
- **VERDICT (pre-registered rule): H2 FALSIFIED.** Both falsifiers fired: (A) CI upper bound of `rr_sonly − rrf` = 0.000 < 0.03; (B) `rr_sonly` = `rr_sonly_shuf` (not better; ≤ holds).
- **Primary:** all_ev@10 `rr_sonly` 0.105 [0.072, 0.141] vs `rrf` 0.105 [0.072, 0.141]; Δ = **0.000** [0.000, 0.000], sign-flip p = 1.0.
  - The all_ev@10 outcome is identical on all 277 questions. The CI is degenerate because every paired difference is 0; it is not evidence of "precisely zero effect of any graph", only that this config (w = 0.1) never flips the 10-turn all-evidence outcome.
  - The graph did change the top-10 turn set on some questions (frac@10 differs from RRF on 5/277 questions; any@10 on 2/277). frac@10: 0.287 vs 0.282, Δ +0.005 [+0.001, +0.010], raw p = 0.063.
  - `rr_sonly_shuf` (seeds 0, 1, 2), `rr_sonly_rand` and `rr_v1W` give 0.105 on all_ev@10 as well. In a 32-question spot check in conversation 1, the shuffled-W graph (5,192 of the W entries differ from real W) produced the same top-10 set as real W on every question.
- Other all_ev@10 (cat 1, ≥ 2 evidence, N = 277), with 95 % CIs:
  - cosine 0.079 [0.051, 0.112]; BM25 0.058 [0.032, 0.087]; RRF 0.105 [0.072, 0.141].
  - `rr_strong` (full W, ζ 0.85, w 0.5) 0.123 [0.087, 0.162]; Δ vs RRF +0.018 [−0.004, +0.043], p = 0.23, Holm 1.0 → **below the 0.03 margin and not significant**.
  - `v1` first stage 0.079 [0.051, 0.112]; Δ vs RRF −0.025 [−0.058, +0.007], p = 0.21, Holm 1.0.
  - `rr_strong − rr_strong_shuf` +0.022 [0.000, 0.043] (p 0.11); `rr_strong − rr_strong_rand` +0.025 [+0.004, +0.051] (p 0.067); `v1 − v1_shuffledW` +0.014 [0.000, 0.033]; `v1 − v1_randomtopo` +0.022 [+0.007, +0.040] (p 0.030 raw, Holm 0.57).
  - Ablations: all S/T/E/A* removals within ±0.011 of the full model, every CI contains 0 (largest: v1_noT +0.011 [−0.011, +0.033], v1_noAstar −0.007 [−0.018, 0.000]). `*_noR` = full model exactly (0.000 [0, 0]) as pre-registered.
  - RRF − BM25 +0.047 [+0.014, +0.079], raw p 0.0065, **Holm-adjusted 0.13**. RRF − cosine +0.025 [−0.007, +0.061], Holm 1.0. On this multi-hop subset RRF is not distinguishable from cosine.
  - Every pairwise comparison in the 20-test pre-registered family is non-significant after Holm (smallest p_Holm = 0.13).
- Secondary metrics (not Holm-adjusted, descriptive): frac@10 `rr_strong − rr_strong_shuf` +0.024 [+0.004, +0.045] (raw p 0.018); `v1 − v1_shuffledW` +0.026 [+0.005, +0.048] (raw p 0.015); `v1 − v1_randomtopo` +0.058 [+0.038, +0.078]; `v1 − rrf` −0.039 [−0.075, −0.002] (raw p 0.039).
  - So on a graded metric, real W may beat shuffled W by ~0.025 on this dataset — the opposite sign from LongMemEval-S (where shuffled was equal or better). It is not corrected for multiplicity, one shuffle seed for those controls, and no single S/T/E ablation reproduces it. I read it as a hint to follow up, not a finding.
- **Prediction check:**
  - ✓ `rr_sonly − rrf` ∈ [−0.01, +0.01] (0.000).
  - ✓ `rr_strong − rrf` ∈ [−0.06, +0.02] (+0.018; point estimate only just inside, CI upper bound +0.043 is outside).
  - ✗ `v1 − rrf` ≤ −0.05 (observed −0.025; the first-stage graph is worse than RRF but by less than predicted, and the CI includes 0).
  - ✓ real ≈ shuffled within ±0.02 for `rr_sonly` and `v1`; ✗ marginally for `rr_strong` (+0.022).
  - ✓ random topology worse than real topology (all three pairs, CIs mostly excluding 0 only uncorrected).
  - ✓ each S/T/E ablation within ±0.02; ✓ noR identical.
  - ✓ H2 falsified (my probability was ~7 % supported).
- Surprises: (1) `rr_sonly` and the shuffled/random/full-W variants are indistinguishable at w = 0.1; the "bonus" is too small to change the 10-set. (2) The strong config gains +0.018, my prior was negative. (3) v1 first stage is closer to cosine than predicted.
- Bugs: attempt 1 aborted (BLAS oversubscription, see above). No metric or retriever bug found; sanity checks passed (noR = full; shuffled W differs from real W entrywise; over all 1,536 questions `locomo_graph` reproduces -02's all_ev@10 exactly: RRF 0.4935, cosine 0.3913).
- Interpretation: on LoCoMo multi-hop, the graph layer does not meet the pre-registered bar. The best graph config (+0.018) is within noise and below the margin; weights are not demonstrably useful; kNN topology is better than random edges (as on LME-S). The absolute headroom is large (RRF gets all evidence in the top-10 for only 10.5 % of multi-hop questions), so the failure is not a ceiling effect: seeds from a weak first stage give the graph little to spread from.
- Intuition update: a graph over MiniLM kNN edges adds at most a small topology benefit. The lever is the first-stage retriever and evidence-aware retrieval (query decomposition), not association weights.
- Decision: no theory version change (nothing in the design changed; v1.1 stays). Ledger updated with L-claims for LoCoMo. Proceed to -04 (QA) as pre-registered.
