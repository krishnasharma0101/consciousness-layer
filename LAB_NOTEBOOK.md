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
