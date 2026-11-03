# STATUS — 2026-11-03 12:10 IST (after R-2026-11-03-04)

Consciousness is not testable here. Φ, Φ_dyn and C_norm are engineering quantities, nothing more.

## What was run (all pre-registered except the v0 backfill; details in LAB_NOTEBOOK.md)

| Run | Result (real numbers, 95% CI) |
|---|---|
| R-2026-10-20-01 (v0, synthetic, n=120) | RAG 113, Sandwich-always 94, shuffled-W 95, λ-trigger 44, no-draft 114, plain 0 |
| R-2026-10-26-01 | MiniLM embedding cache, 190,015 texts, 1,073 s |
| R-2026-10-26-02 (E3, LongMemEval-S, n=470) | turn_any@10: BM25 0.857, cosine 0.855, v1 graph 0.819. v1−cos −0.036 [−0.062, −0.011], Holm 0.055. Random topology −0.098 vs cos. Removing S/T/E raises nDCG. |
| R-2026-10-26-03 (E2) | AUROC(−C_norm → failure): 0.63–0.68 on LongMemEval, below max-cosine for v1/cosine misses (0.72, 0.74) and slightly above it for BM25 misses (0.625 vs 0.603); synthetic 0.671 [0.556, 0.782]. Residual 0.56–0.59; within probe type 0.576 [0.393, 0.754] |
| R-2026-10-26-04 (E3b, test n=247) | Graph re-ranking bonus: Δ 0.000 [−0.016, 0.016] (cosine), 0.000 [0, 0] (RRF). Exploratory: RRF(BM25, cosine) 0.913 vs cosine 0.855, +0.057 [0.028, 0.087] |
| R-2026-11-03-01/02 (LoCoMo, N=1,536) | all_ev@10: cosine 0.391, BM25 0.467, RRF 0.493 [0.469, 0.518]; RRF−cosine +0.102 [0.081, 0.124], RRF−BM25 +0.026 [0.007, 0.045] (Holm 0.010; clustered CI includes 0). H1 confirmed |
| R-2026-11-03-03 (LoCoMo multi-hop, N=277) | H2 **falsified**: S-only graph rerank − RRF 0.000 [0, 0]; full-W rerank +0.018 [−0.004, 0.043]; v1 first stage −0.025 [−0.058, 0.007]; ablations/shuffled/random all within ±0.02; nothing survives Holm |
| R-2026-11-03-04 (QA, Qwen2.5-1.5B, N=277) | F1: RRF ctx 0.258, graph ctx 0.258 (Δ −0.0006 [−0.015, 0.013]), Sandwich no-draft 0.220, with draft 0.229, closed-book 0.145 |
| (R-2026-11-03-03 attempt 1) | aborted: BLAS thread oversubscription; logged |
| E5/E6 | skipped (reason logged): synthetic is saturated; end-to-end QA needs a paid judge or a GPU |

## What the theory is now (docs/THEORY_V1.md v1.1)
- **Well-defined and bounded (math and tests):** W, A (personalised PageRank), C_norm, Φ_dyn, admissible A*.
- **Empirically on LongMemEval-S:**
  - The graph does not beat plain retrieval as a first stage, as a weighting, or as a re-ranking bonus.
  - Only its kNN topology beats random edges.
  - C_norm as a failure signal is weak and mostly restates retrieval confidence.
  - The usable part of the Sandwich is "retrieve well (RRF), then have a second LLM answer without anchoring on the draft" (v0 no-draft 114/120, exploratory).

## Ledger (docs/THEORY_LEDGER.md, 43 claims)
supported 6 · refuted 19 · mixed 5 · weak evidence 3 · untested 3 · not testable 7

## Tests
`pytest`: 52 passed (re-run 2026-11-03).

## Update 2026-11-03
- LoCoMo replicates the LongMemEval-S picture: RRF(BM25, cosine) is the strong baseline; the graph layer (as bonus or first stage) adds nothing that survives correction; W values carry at most a tiny uncorrected signal; kNN topology beats random edges weakly.
- End-to-end with a 1.5B reader: graph context = RRF context, Sandwich retrieval no better. THEORY_V1 stays at v1.1 (no design change).
- Paper: paper/ (see paper/SUBMISSION_NOTES.md).

## Needs Krishna's decision
1. **End-to-end QA judge.**
   - The official LongMemEval judge is GPT-4o, which is paid.
   - The alternatives are a free local judge (weak on CPU) or a GPU. Approve one, or accept retrieval-only evidence.
2. **GPU time** (e.g. Colab/Kaggle free tier, but that needs your account) for an LLM reader at ≥ 7B.
3. (Done in part: LoCoMo multi-hop was run, negative.) **A multi-hop memory dataset** where associations could matter (the graph's last plausible niche). Options include a multi-hop subset of LoCoMo or similar; check its licence and access.
4. **Paper framing.** Given these results, the honest framing is "a bounded graph formalism; empirically no gain over RRF retrieval on LongMemEval-S", not "a framework for consciousness".

## Files
docs/{THEORY_V1.md, THEORY_LEDGER.md, STATUS.md, MATH_AUDIT.md, EXPERIMENT_PLAN.md} · LAB_NOTEBOOK.md · EXPERIMENTS.md ·
src/conslayer/{theory_v1.py, runlog.py, stats.py} · tests/test_v1.py · experiments/{e3_lme_retrieval.py, e2_cnorm_signal.py,
e2_posthoc_within_ptype.py, e3b_hybrid.py, lme_common.py, lme_embed.py} · results/<run_id>/ (config.json + CSVs)
