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
