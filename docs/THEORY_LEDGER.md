# THEORY LEDGER

Every claim in the paper (P-*) and in THEORY_V1 (V-*), with its current evidential status.
Statuses: **untested** · **supported (run IDs)** · **refuted (run IDs)** · **not testable (why)** ·
**mixed** (supported in one setting, refuted in another). "Synthetic" = v0 40-dialogue benchmark, which is
saturated for retrieval and small; treat synthetic results as weak evidence.
Updated after every run; the change log at the bottom records what changed and why.

## Paper claims

| ID | Claim (section) | Status | Evidence / note |
|---|---|---|---|
| P-01 | Consciousness is a specific mathematical/structural process (§1) | not testable | No operational test for consciousness exists; the framework only defines graph quantities. |
| P-02 | The framework is "testable … for instantiating synthetic consciousness" (§1) | not testable | Same; only engineering sub-claims below are testable. |
| P-03 | An external graph memory supplies cross-turn statefulness missing from a context-limited LLM (§2.1, §5.2) | supported (R-2026-10-20-01) | Synthetic: Sandwich 94/120 vs plain 0/120. Says nothing about graph vs. simpler memory (P-04). |
| P-04 | The graph memory is better than plain vector retrieval (implicit in §2.2, §6) | refuted (R-2026-10-20-01, R-2026-10-26-02) | Synthetic: RAG 113/120 > Sandwich 94/120. LongMemEval-S (n=470) turn_any@10: v1 0.819 vs cosine 0.855, Δ −0.036 [−0.062, −0.011], sign-flip p 0.009, Holm 0.055 (borderline); BM25 0.857. Worse at @1 (0.164 vs 0.353) and nDCG@10 (0.420 vs 0.522). Caveat: MiniLM truncates 34% of turns at 256 word-pieces. |
| P-05 | W = S·T·(1+E)·R carries useful structure (Eq. 1) | refuted for the weights; topology matters (R-2026-10-20-01, R-2026-10-26-02) | Shuffled W vs v1: recall@10 +0.006 [−0.015, 0.028], nDCG@10 +0.040 [0.026, 0.055] (shuffled is *better*). Random topology: recall@10 −0.062 [−0.085, −0.038]. So which pairs are linked (top-k by τ) matters; the W values on them do not help. |
| P-06 | Semantic similarity S is a useful association term (Eq. 2) | refuted as a W factor (R-2026-10-26-02) | Removing S from W: recall@10 +0.000 [−0.011, 0.011], nDCG@10 +0.012 [0.004, 0.020] (removal helps). S still defines the edges via τ and the seeds; only its role inside W is refuted. |
| P-07 | Temporal closeness T is a useful association term (Eq. 3) | refuted on LongMemEval-S (R-2026-10-26-02) | Removing T: recall@10 +0.006 [−0.006, 0.019], nDCG@10 +0.014 [0.008, 0.021]. h = 7 days was not tuned. |
| P-08 | Emotion E is "the heuristic driving the routing of thought" / multiplies significance (§2.1, Eq. 1) | refuted on LongMemEval-S (R-2026-10-26-02) | Removing E (lexicon arousal): recall@10 +0.015 [0.000, 0.030], nDCG@10 +0.021 [0.011, 0.032]. Caveat: lexicon valence is crude, and LongMemEval is not an emotional task. |
| P-09 | Hebbian reinforcement R improves behaviour with use (Eq. 4) | untested | E3 cannot test (no learning ⇒ R≡1, see THEORY_V1 §4). Needs E9 (multi-session). |
| P-10 | (1+E_ij) is a valid multiplier (Eq. 1) | refuted (math, MATH_AUDIT #8) | E ∈ [−1,1] gives zero/negative weights; v1 uses 1+κ_E·arousal. |
| P-11 | Conscious thought = A* optimal path with an emotion+logic heuristic (§3.2) | mixed (math ✓; usefulness ✗ for recall) | Optimality holds for admissible H (tests). Usefulness (R-2026-10-26-02): removing A* changes recall@10 by −0.002 [−0.006, 0.000] but nDCG@10 by −0.018 [−0.024, −0.013], so the path only reorders the top-10 a little. start ≠ goal in 66.8% of questions; goal is a gold turn in 16.4%. |
| P-12 | Random-walk "subconscious" surfaces useful associations (Eq. 6) | refuted as first stage (R-2026-10-26-02) and as re-ranking bonus (R-2026-10-26-04) | Ranking by PPR activation (ζ = 0.85) is worse than ranking by the seed similarity itself (v1 vs cosine above). Less diffusion helps: ζ 0.5 nDCG +0.048, ζ 0.15 +0.071 vs ζ 0.85, i.e. the closer to plain cosine, the better. Multi-session (where associations should help most): v1 0.760 vs cosine 0.868. |
| P-13 | High C(t) means mutually supportive concepts are active (Eq. 7) | refuted on synthetic (R-2026-10-20-01, exploratory) | C_norm was highest on update probes (0.142 vs 0.039–0.046), i.e. when *conflicting* facts were co-active. |
| P-14 | Φ (Eq. 8) measures depth of consciousness at a moment | not testable / refuted as a dynamic quantity | Consciousness: not testable. Eq. 8 has no time dependence (MATH_AUDIT #1). v1 Φ_dyn equals C_norm at convergence in every LongMemEval question (R-2026-10-26-03), so it adds nothing. |
| P-15 | Φ is a tractable analogue of IIT Φ (§2.3) | refuted (math, MATH_AUDIT #1) | Eq. 8 is a weighted degree sum; no partition/integration term. |
| P-16 | τ_ij governs useful formation of new connections (Eq. 9) | untested | E4 θ sweep / edge-density ablation. |
| P-17 | C(t) > λ ⇒ self-awareness (Eq. 10) | not testable | No observable for self-awareness; v1 keeps λ only as a refine trigger. |
| P-18 | C(t) < λ ⇒ primary response insufficient (§5.2) | weak evidence, mostly confounded (R-2026-10-26-03) | AUROC(−C_norm → failure). Synthetic Sandwich-always: 0.671 [0.556, 0.782], but within contradiction probes only (post-hoc) 0.576 [0.393, 0.754], so it is mostly a probe-type confound. LongMemEval v1 misses: 0.653 [0.584, 0.719], below max cosine alone 0.719; residual after maxcos 0.564 [0.495, 0.630]. |
| P-19 | Triggering refinement only when C < λ helps (§5.2) | refuted on synthetic (R-2026-10-20-01) | λ-trigger 44/120 vs always-refine 94/120 (λ = 0.0532, chosen post hoc). |
| P-20 | Secondary LLM refinement improves the answer (§5.3) | mixed (R-2026-10-20-01) | Over plain: 94 vs 0. Over RAG: draft-in-prompt 94 < RAG 113; no-draft prompt 114 ≈ RAG 113 (exploratory). Self-refine without memory 0/120. |
| P-21 | Secondary LLM updates W directly (§5.3) | not testable as stated | Unspecified mechanism (MATH_AUDIT #15); v1 replaces with success signal s. |
| P-22 | Outputs become emotionally weighted, creative, non-repetitive (§6) | untested | No metric defined. |
| P-23 | Path to AGI / self-aware entities (§7) | not testable | Rhetorical. |
| P-24 | EEG/BCI-controlled hyper-parameters (§6 future) | not testable here | Out of scope (no hardware). |

## THEORY_V1 claims

| ID | Claim | Status | Evidence |
|---|---|---|---|
| V-01 | All W, T, F, R, C_norm, Φ_dyn bounded as specified | supported (unit tests, 3aec799) | `tests/test_v1.py` |
| V-02 | Activation update converges to a unique fixed point (PPR) | supported (math + unit test) | ζ<1 contraction; `test_activation_fixed_point` |
| V-03 | H is admissible ⇒ A* returns the min-cost path | supported (math + unit test) | A* == Dijkstra on random graphs |
| V-04 | v1 graph retrieval recall@10 ≥ cosine on LongMemEval-S | refuted (R-2026-10-26-02) | Δ −0.036 [−0.062, −0.011], p 0.009, Holm 0.055. |
| V-05 | Real W beats shuffled-W and random-topology controls | mixed: ✓ vs random topology, ✗ vs shuffled W (R-2026-10-26-02) | See P-05. |
| V-06 | Each of S, T, E contributes (ablation lowers recall) | refuted (R-2026-10-26-02) | No ablation lowered recall@10; all three removals raised nDCG@10 (CIs > 0). |
| V-07 | A* path adds recall over ranking by activation alone | refuted for recall, small nDCG gain (R-2026-10-26-02) | See P-11. |
| V-08 | Low C_norm predicts wrong answers / retrieval misses (AUROC > 0.5) | weak evidence; mostly retrieval confidence (R-2026-10-26-03) | AUROC > 0.5 everywhere, but below or equal to max-cosine, and residual CIs touch 0.5. See P-18. |
| V-09 | Secondary prompt without the draft avoids anchoring on a wrong draft | supported on synthetic (R-2026-10-20-01, exploratory) | 114/120 vs 94/120; not pre-registered. |
| V-10 | Recency-weighted input helps knowledge-update questions | refuted (R-2026-10-26-02, exploratory) | Knowledge-update questions: recency 0.889 vs plain v1 0.944 recall@10; overall nDCG −0.017 [−0.028, −0.006]. |
| V-11 | Every −log W path cost is invariant to rescaling R (THEORY_V1 v1.0 §4) | refuted (spec error, R-2026-10-26-02) | The −log cost adds a per-hop penalty of log W_max; 11/470 paths changed. Corrected in v1.1. Invariance holds for P and A only. |
| V-12 | Graph activation as a re-ranking bonus on a first stage improves recall@10 (v1.2 candidate) | refuted (R-2026-10-26-04) | Test half n=247, dev-chosen config: cosine Δ 0.000 [−0.016, 0.016]; RRF Δ 0.000 [0.000, 0.000]; Holm 1.0 both; shuffled W identical recall; larger w hurts (RRF w0.5 ζ0.85: 0.874 vs 0.907). |

## Counts (36 claims: 24 paper, 12 v1)

| Status | Count |
|---|---|
| supported | 5 (P-03, V-01, V-02, V-03, V-09) |
| refuted (empirical, math, or spec error) | 16 (P-04, P-05, P-06, P-07, P-08, P-10, P-12, P-13, P-15, P-19, V-04, V-06, V-07, V-10, V-11, V-12) |
| mixed | 3 (P-11, P-20, V-05) |
| weak evidence | 2 (P-18, V-08) |
| untested | 3 (P-09, P-16, P-22) |
| not testable | 7 (P-01, P-02, P-14, P-17, P-21, P-23, P-24) |

## Change log

| Date (IST) | Change | Run |
|---|---|---|
| 2026-10-26 | Ledger created; statuses from v0 backfill and MATH_AUDIT | R-2026-10-20-01 |
| 2026-10-26 | P-04, V-04 refuted on LongMemEval-S. P-05 refuted for weights (topology matters), V-05 mixed. P-06/07/08, V-06 refuted (removing S/T/E raises nDCG). P-11 usefulness ✗ for recall; V-07 refuted. P-12 refuted as first stage. V-10 refuted. V-11 added (spec error, fixed in THEORY_V1 v1.1). V-12 added (pre-registered hybrid). Counts redone; P-14 now counted under not testable only. | R-2026-10-26-02 |
| 2026-10-26 | P-18, V-08 → weak evidence (AUROC > 0.5, but explained mostly by probe type / max-cosine). P-14: Φ_dyn = C_norm at convergence. | R-2026-10-26-03 |
| 2026-10-26 | V-12 refuted (graph bonus adds 0.000 on held-out half); P-12 extended to "refuted as bonus". Not a ledger claim, but noted: RRF(BM25, cosine) 0.913 beats both singles by +0.055–0.057 (exploratory). | R-2026-10-26-04 |
