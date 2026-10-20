# EXPERIMENT_PLAN — testing the Consciousness Layer as an LLM memory mechanism

**What these experiments can and cannot show.** They can show whether the graph layer (Eqs. 1–10,
with the MATH_AUDIT fixes) helps an LLM system *behave* better on memory, consistency and answer
quality than simpler baselines. They cannot show consciousness, self-awareness or "depth of
consciousness": no behavioural or numerical result distinguishes a conscious system from one that
merely computes the same function, and Φ/C here are not validated consciousness measures. Results
should be written up as "a graph-based memory layer for LLMs", with the consciousness framing at most
as motivation.

## Hypotheses (all falsifiable, all behavioural)

- **H1 (memory recall).** Sandwich-graph retrieval answers questions about facts stated earlier in a
  conversation more accurately than (a) a plain LLM with the same recent window and (c) vector-memory
  RAG with the same retrieval budget k.
- **H2 (consistency / knowledge update).** It handles updated facts and user-asserted contradictions
  better than (a), (b) self-refine without memory, and (c).
- **H3 (each term matters).** Removing any one of S, T, E, R, A*, random walk/spreading activation,
  or the λ trigger lowers performance; a shuffled-weight or random graph is worse than the real graph.
  (If the graph does no better than shuffled weights, the weighting formula carries no information.)
- **H4 (C as a signal).** C_norm predicts when the draft answer is wrong (AUROC > 0.5 with a CI that
  excludes 0.5), so the λ trigger saves LLM calls at equal accuracy.

**Baselines (always same LLM, same decoding, same recent window, same k):**
(a) plain LLM (= the Primary LLM alone); (b) Primary + self-refine pass, no memory (Madaan et al.
2023, arXiv:2303.17651); (c) vector RAG: top-k cosine over stored turns (MiniLM); plus a
full-context reference (whole history in the prompt) where it fits, as an upper bound, and
sandwich-always (λ = ∞) to separate the retrieval from the trigger.

**Statistics.** Paired design (all systems see the same items). Report counts, accuracy, 95% CI
(Wilson for proportions; paired bootstrap for differences) and McNemar's exact test for pairwise
comparisons. ≥ 3 seeds for synthetic data. Pre-register the primary metric per experiment (below)
before running; tune λ, θ, γ, k only on a dev split.

## Benchmarks (verified 2026-10-20)

| Benchmark | What | Size | Link | Notes |
|---|---|---|---|---|
| LongMemEval (ICLR 2025) | chat-assistant long-term memory: info extraction, multi-session, knowledge update, temporal, abstention | 500 questions; `_S` ≈ 115k tokens history (~40 sessions)/question; `_M` ≈ 500 sessions; `oracle` = evidence sessions only | https://github.com/xiaowu0162/longmemeval , data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned | turn-level `has_answer` labels ⇒ retrieval recall can be scored **without any LLM**. Official QA metric uses an LLM judge (GPT-4o in the repo) — paid. |
| LoCoMo (ACL 2024) | very long-term conversations; QA (single-hop, multi-hop, temporal, commonsense, adversarial), event summarisation | 10 conversations (`data/locomo10.json`), ~300 turns / ~9K tokens avg per project page | https://github.com/snap-research/locomo | QA scored with token F1 (`task_eval/evaluation.py`) ⇒ no judge needed; evidence dialog ids ⇒ retrieval recall. |
| MSC — Multi-Session Chat ("Beyond Goldfish Memory", Xu et al. 2021, arXiv:2107.07567) | human–human chats over up to 5 sessions with persona summaries | HF mirror 23,445 rows | https://parl.ai/projects/msc/ , https://huggingface.co/datasets/nayohan/multi_session_chat | No memory QA labels; use for persona-consistency (needs an NLI model or judge). Lower priority. |
| Synthetic (this repo) | facts across turns, distractors, one fact updated, probes: recall / update / contradiction | configurable; v0 = 40 dialogues × 3 probes | `src/conslayer/synthetic.py` | Fully controlled, exact-match scoring, cheap. Risk: too easy / templated; results don't transfer automatically. |

## Compute baseline (measured on this box, see EXPERIMENTS.md)

8 vCPU Xeon (AVX-512), 15 GiB RAM, no GPU. Qwen2.5-1.5B-Instruct (HF transformers, bf16, CPU):
short-prompt answers ~0.5–3 s per call in v0; Qwen2.5-0.5B ~2× faster. MiniLM embeddings are
negligible. All wall-clock numbers below marked *est.* are extrapolations from these v0
measurements, not measurements. Long prompts (1–4k tokens of retrieved context) will be much slower
on CPU because prefill dominates — budget 5–20 s per call (*est.*) until measured.

## Prioritised experiments

| # | Name | Hypothesis | CPU-feasible? | Est. wall-clock (CPU) |
|---|------|-----------|---------------|------------------------|
| E0 | Maths/unit sanity | implementations match the equations & audit claims | yes | done (seconds) |
| E1 | Synthetic v0 comparison | H1, H2 (pilot) | yes (1.5B) | measured: see EXPERIMENTS.md (~15 min) |
| E2 | C_norm as error signal (from E1/E3 logs) | H4 | yes, no new LLM calls | minutes |
| E3 | LongMemEval retrieval recall (no LLM) | graph retrieval recall@k ≥ vector RAG | yes | *est.* 0.5–2 h for 500 Q (embedding + graph build per haystack; graph build O(N·k)) |
| E4 | Ablations + controls on E3 & synthetic | H3 | yes | *est.* E3-time × #configs (~15 configs) ⇒ several hours; synthetic ~15 min/config |
| E5 | Synthetic scaled: 200 dialogues × 3 seeds, longer histories (50–200 turns), 0.5B/1.5B/3B | H1, H2 with power | yes (3B slow) | *est.* 1.5B: 4–6 h; 3B: 2–3× that |
| E6 | λ trigger sweep (below/above/always/never, λ quantiles) | H4: same accuracy, fewer calls | yes | *est.* ~1 h on synthetic |
| E7 | LoCoMo QA (token F1) | H1 on real data | partly: 1.5B, retrieval-only (k turns), all 10 conversations | *est.* 3–8 h (long prompts) |
| E8 | LongMemEval QA (subset 100 Q, then 500) | H1/H2 on real data | subset yes with substring/F1 or local judge; official metric needs paid judge | *est.* 100 Q × 5 systems: 3–6 h; 500 Q: 1–2 days |
| E9 | Learning dynamics of R | R updates help on *disjoint* later queries; weight concentration bounded | yes | *est.* 2–4 h |
| E10 | Emotion term relevance | E helps on emotionally salient facts, hurts nothing else | yes (synthetic with emotional/neutral facts) | *est.* 1 h |
| E11 | MSC persona consistency | fewer persona contradictions | NLI model on CPU OK; judge LLM better on GPU/API | *est.* 4–8 h |

### E1 — Synthetic v0 (run)
- **Setup.** 40 dialogues (seed 0), 15 user turns each: 5 facts + 9 distractors + 1 fact update; the
  last 4 turns (the recent window every system sees) are always distractors. Probes per dialogue:
  recall, update (new value present, old absent), contradiction ("my X is <wrong>, right?" — correct
  iff the true value is stated). λ = median C_norm on a disjoint dev set (10 dialogues, seed 1).
- **Metrics.** exact substring accuracy (counts), latency per probe, LLM calls.
- **Falsifier.** Sandwich ≤ vector RAG on all probe types ⇒ no evidence for H1/H2 beyond plain RAG.
  Sandwich ≈ sandwich-shuffled ⇒ the weights don't matter (H3 fails at this scale).
- **Limits.** 120 probes ⇒ 95% CI width ≈ ±9 pp near 50% accuracy; templated sentences; one model.

### E2 — Is C_norm informative? (free)
- **Hypothesis.** P(draft wrong | C_norm low) > P(draft wrong | C_norm high).
- **Metric.** AUROC of −C_norm for "plain draft is wrong"; also AUROC for "sandwich answer wrong".
- **Falsifier.** AUROC CI includes 0.5 ⇒ the λ trigger is a coin flip; drop it or redefine C.

### E3 — LongMemEval retrieval recall (top CPU priority after E1)
- **Setup.** For each of the 500 questions in `longmemeval_s_cleaned.json`: nodes = turns of the
  haystack sessions (timestamps from `haystack_dates`); retrieve k ∈ {5, 10} turns with (c) cosine,
  (g) sandwich graph retrieval (spreading activation + A*), (g′) shuffled weights, (g″) random graph
  with matched degree, plus BM25 as a second classic baseline.
- **Metrics.** turn-level recall@k and session-level recall@k against `has_answer` /
  `answer_session_ids`, per question type. NDCG@k.
- **Falsifier.** graph recall@k ≤ cosine recall@k (paired bootstrap CI overlaps 0 or negative) ⇒ the
  graph adds nothing as a retriever on real data.
- **Compute.** No LLM. Embedding ~115k tokens/question with MiniLM on 8 cores and an O(N·k) graph per
  question; *est.* 0.5–2 h total. Needs ~ a few GB disk for the data.

### E4 — Ablations and controls
- One-at-a-time: `use_S/use_T/use_E/use_R=False`; A* off (activation only); spreading activation off
  (A* only); emotion `paper` vs `arousal` vs `exp`; cost `inverse` vs `neglog`; θ sweep; direction
  `symmetric` vs `forward`; shuffled weights; random graph (same N, same degree sequence).
- Metric: E3 recall@k (primary), synthetic accuracy (secondary).
- Falsifier per term: removing it doesn't hurt (CI includes 0) ⇒ the term is unnecessary; say so.

### E5 — Scaled synthetic (power)
- 600 probes per seed × 3 seeds; harder variants: more distractors (window → history ratio), multi-hop
  facts ("my sister lives in the city where I studied"), multiple updates, paraphrased probes.
- Models: Qwen2.5-0.5B / 1.5B / 3B-Instruct (CPU). With ~1,800 paired probes a 5 pp difference is
  detectable (*est.*, depends on discordant-pair rate).

### E6 — λ trigger
- Cost–quality curve: accuracy vs mean LLM calls per probe for λ quantiles {0.1,…,0.9}, direction
  below vs above. Falsifier: "below" is not on the Pareto front vs simply always/never refining.

### E7/E8 — Real QA (LoCoMo, LongMemEval)
- Systems (a), (b), (c), sandwich, sandwich-shuffled, full-context where it fits in the model context.
- LoCoMo: official token-F1 per category (no judge). LongMemEval: official LLM-judge metric is the
  comparable number; CPU-only alternatives are substring match / local judge, reported as such
  and not compared with published numbers.
- Paid judge cost (NOT spent): 500 Q × ~5 systems = 2,500 short judge calls; with a small hosted
  model this is in the order of a few USD, with GPT-4o-class models roughly tens of USD (*est.* —
  check current pricing before running). A free alternative is a local 7–8B judge on a GPU.

### E9 — R learning dynamics
- Stream 500+ queries over one long history; update R with (i) label-free proxy, (ii) oracle success on
  a train split only. Evaluate on disjoint test queries. Track Gini of W and accuracy over time.
- Falsifier: proxy-R ≤ R off ⇒ Eq. (4) with self-assessed success is not useful (or harmful).

### E10 — Emotion term
- Synthetic facts tagged emotional vs neutral; is recall of emotional facts improved by E without
  hurting neutral ones? Falsifier: no difference, or neutral recall drops more than emotional rises.

### E11 — MSC persona consistency
- Contradiction rate of model turns vs persona summaries, scored by an NLI model (e.g. a DeBERTa MNLI
  checkpoint on CPU). Lowest priority: noisy metric, no QA labels.

## GPU / paid needs (nothing has been spent)

- Everything in E0–E7, E9, E10 runs on this CPU box with ≤3B models (slowly).
- GPU helps for: 7–8B models (to check the effect holds beyond toy-size models), full LongMemEval
  QA (500 Q × systems with long prompts), local judging. A single 24 GB GPU is sufficient; free
  options are Kaggle/Colab free tiers (time-limited).
- Paid API only for the official LongMemEval GPT-4o judge (see E8 estimate) — optional.

## What is publishable, honestly

- **Possibly publishable (workshop / arXiv):** "A graph-weighted memory layer for LLM conversation
  memory: equations, fixes, and an ablation study" — *if* E3/E4 show a statistically reliable gain
  over cosine RAG and BM25 on LongMemEval/LoCoMo, with ablations showing which terms matter, and if it
  is honestly positioned against Generative Agents (arXiv:2304.03442), HippoRAG (arXiv:2405.14831),
  MemGPT (arXiv:2310.08560), A-MEM and similar. A clean *negative* result (graph ≈ RAG; emotion term
  useless; C not predictive) with the audit is also an acceptable workshop/arXiv contribution.
  Suitable venues: memory/agents or LLM-reasoning workshops at ACL/EMNLP/NeurIPS/ICLR; arXiv cs.CL/cs.AI.
- **Not publishable as is:** any claim of consciousness, self-awareness, AGI, or that Φ here relates
  to IIT; results only on the in-house synthetic task; results without baselines (b)/(c) and the
  shuffled-weights control; numbers from a single seed with ~100 items; the current paper's
  framing (no experiments, contradictory λ, undefined H/B/success).
