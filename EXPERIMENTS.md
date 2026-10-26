# EXPERIMENTS log

Every run gets an entry. Real numbers only; copy from `results/<run>/` files, never retype from memory.

## Index (one row per run; details in LAB_NOTEBOOK.md, artifacts in results/<run>/)

| Run ID | Date (IST) | What | Pre-registered | Headline result | Notebook | Results |
|---|---|---|---|---|---|---|
| R-2026-10-20-01 | 2026-10-20 | E1/v0 synthetic Sandwich vs baselines (entry below) | no (backfill) | RAG 113/120, Sandwich-always 94/120, shuffled 95/120, no-draft 114/120 | LAB_NOTEBOOK.md#r-2026-10-20-01 | results/v0/ |
| R-2026-10-26-01 | 2026-10-26 | LongMemEval-S MiniLM embedding cache (preprocessing) | yes | 190,015 texts embedded in 1,073 s; sha256 684ef8b5…cfb8 | LAB_NOTEBOOK.md#r-2026-10-26-01 | results/R-2026-10-26-01/ |
| R-2026-10-26-02s | 2026-10-26 | E3 smoke test (few questions, code check only) | yes | smoke only, no conclusions | LAB_NOTEBOOK.md#r-2026-10-26-02s | — |
| R-2026-10-26-02 | 2026-10-26 | E3 LongMemEval-S retrieval: v1 graph vs BM25/cosine + controls/ablations (n=470) | yes | turn_any@10: cosine 0.855, BM25 0.857, v1 0.819; v1−cos −0.036 [−0.062, −0.011], Holm 0.055; random topology −0.098; S/T/E removal raises nDCG | LAB_NOTEBOOK.md#r-2026-10-26-02 | results/R-2026-10-26-02/ |
| R-2026-10-26-03 | 2026-10-26 | E2 C_norm as a failure predictor (synthetic + LongMemEval) | yes (no primary label named; logged flaw) | AUROC 0.65–0.67, ≤ max-cosine; residual 0.56–0.59; within probe type 0.576 [0.393, 0.754] | LAB_NOTEBOOK.md#r-2026-10-26-03 | results/R-2026-10-26-03/ |
| R-2026-10-26-04 | 2026-10-26 | E3b graph activation as re-ranking bonus on cosine / RRF first stage (dev/test split) | yes | test Δ turn_any@10: cosine 0.000 [−0.016, 0.016], RRF 0.000 [0, 0]; Holm 1.0; exploratory: RRF alone 0.913 vs cosine 0.855 (+0.057 [0.028, 0.087]) | LAB_NOTEBOOK.md#r-2026-10-26-04 | results/R-2026-10-26-04/ |
| (E5/E6) | 2026-10-26 | skipped, reason logged before running | — | synthetic saturated; end-to-end QA needs paid judge or GPU | LAB_NOTEBOOK.md#e5e6--skipped | — |

## Template

```
### <ID> — <name>   (date, IST)
Git commit:
Command:
Config: model / embedder / decoding / window / k / lambda (+ how calibrated) / graph WeightConfig
Seed(s): task seed, dev seed
Data: dataset + split + n items
Hardware: box (8 vCPU, 15 GiB, no GPU) / other
Wall clock:
Results table: system | correct / n | acc | 95% CI | mean latency | LLM calls
Significance: McNemar exact p (paired) vs RAG and vs plain
Plots: results/<run>/*.png
Diagnostics:
Limitations:
Conclusion (one paragraph, no hype):
```

---

### E1 / v0 — synthetic multi-turn memory, 4 baselines + Sandwich variants   (2026-10-20, IST)

- **Command:** `.venv/bin/python experiments/run_v0.py --n 40 --seed 0 --dev-n 10 --dev-seed 1 --out results/v0`
  then `experiments/analyze_v0.py`, `experiments/retrieval_diag_v0.py`, `experiments/nodraft_v0.py`.
- **Config:** Qwen/Qwen2.5-1.5B-Instruct (HF transformers 5.2, bf16, CPU, greedy, max 48/60 new tokens);
  embedder sentence-transformers/all-MiniLM-L6-v2; recent window = last 4 user turns (always
  distractors by construction); memory budget k = 4 turns for every memory system;
  WeightConfig defaults (audit fixes): γ=0.05/turn, emotion=arousal (1+|e|), R bounded (η=0.1,
  R∈[1,2]), W globally normalised, symmetric, τ: α=0.2, β=0.7, ρ=0.1, θ=0.2, max out-degree 8;
  activation ζ=0.85, 20 iterations, 5 seed nodes; A* heuristic blend_scaled (μ=0.7), cost 1/(W+ε).
  λ trigger direction = "below" (refine iff C_norm < λ, as in §5.2); λ = median C_norm on a disjoint
  dev set (10 dialogues, seed 1, 30 probes) = **0.0532**. Success signal for Eq. (4) = label-free
  proxy (final answer cos-sim > 0.5 to a retrieved memory).
- **Data:** synthetic, 40 dialogues × 15 user turns, 3 probes each (recall / update / contradiction) = 120 probes.
- **Hardware / wall clock:** box CPU; 1,092 s for the main run (814 LLM calls incl. dev + warm-up);
  no-draft follow-up 139 s.
- **Latency** = wall time per probe including the primary call for two-stage systems.

| system | correct / 120 | acc | 95% Wilson CI | recall /40 | update /40 | contradiction /40 | mean latency s | LLM calls/probe |
|---|---|---|---|---|---|---|---|---|
| (a) plain (= Primary LLM, window only) | 0 | 0.000 | 0.000–0.031 | 0 | 0 | 0 | 1.00 | 1 |
| (b) self-refine, no memory | 0 | 0.000 | 0.000–0.031 | 0 | 0 | 0 | 2.60 | 2 |
| (c) vector RAG (top-4 cosine) | 113 | 0.942 | 0.884–0.971 | 39 | 40 | 34 | 1.03 | 1 |
| Sandwich, λ trigger (below) | 44 | 0.367 | 0.286–0.456 | 27 | 2 | 15 | 1.58 | 1.52 |
| Sandwich, always refine | 94 | 0.783 | 0.701–0.848 | 37 | 38 | 19 | 2.07 | 2 |
| Sandwich, shuffled weights (control) | 95 | 0.792 | 0.711–0.855 | 36 | 39 | 20 | 2.13 | 2 |
| Sandwich retrieval + RAG prompt (no draft) — follow-up | 114 | 0.950 | — | 39 | 40 | 35 | 1.08 + primary | 2 |
| reference: full history in context | 114 | 0.950 | 0.895–0.977 | 39 | 39 | 36 | 1.22 | 1 |

(CI for the follow-up row not computed by the analysis script; 114/120 gives the same Wilson CI as the full-context row.)

- **McNemar exact vs RAG** (`results/v0/mcnemar.csv`): Sandwich-always 1 vs 20 discordant, p < 0.001;
  Sandwich-shuffled 1 vs 19, p < 0.001; Sandwich-λ 1 vs 70, p < 0.001; full-context 4 vs 3, p = 1.0.
- **Retrieval diagnostic (no LLM, `retrieval_recall.csv`):** the gold value is inside the k=4 retrieved
  turns in **120/120** probes for RAG, Sandwich (default), Sandwich with shuffled weights, no-T, no-E,
  and S-only. The task does not discriminate between retrievers at all.
- **λ trigger diagnostic:** refine rate 51.7%; refined → 44/62 correct, not refined → 0/58 correct
  (the primary never has the facts in its window, so skipping refinement is always wrong).
  Mean C_norm is highest on update probes (0.142 vs 0.039–0.046), so update probes were almost never
  refined (5%) → 2/40. Old and new values of an updated fact are near-duplicates with a strong edge, so
  the conflict *raises* C (MATH_AUDIT item 18 observed in practice).
- **C_norm as an error signal (`signal.json`):** AUROC of low C_norm for "Sandwich-always answer is
  wrong" = 0.659 (n = 120, 26 wrong; no CI computed). For "plain draft is wrong" it is undefined (the
  draft was wrong in 120/120).
- **Plot:** `results/v0/accuracy.png`.

**Conclusion.** On this task the graph layer adds nothing over plain vector RAG, and the full
Sandwich pipeline is worse (94 vs 113 of 120). The deficit is not retrieval (all retrievers recall the
fact 120/120; Sandwich retrieval with the plain RAG prompt scores 114/120): it comes from feeding the
Primary LLM's wrong draft to the Secondary LLM, which anchors it — mostly on contradiction probes
(19/40 vs 34/40), where the draft already agreed with the user's false claim. Shuffling the graph
weights changes nothing (95 vs 94), so the weighting formula carried no measurable information here.
The λ trigger as specified (refine when C < λ) hurts, because C is high exactly when contradictory
memories coexist. Self-refine without memory cannot recover facts it never saw (0/120).

**Limitations.** 120 probes, one seed, one 1.5B model, templated English sentences, exact-substring
scoring (lenient for answers that mention both values on non-update probes), 15-turn histories where
k=4 retrieval is trivially sufficient. The graph's possible advantages (multi-hop, long histories,
learning via R) are not exercised by this task. These results say the v0 design choices
(draft-in-prompt, low-C trigger) are bad; they do not say the graph can never help — that needs E3–E5.
