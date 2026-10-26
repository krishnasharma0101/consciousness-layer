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
