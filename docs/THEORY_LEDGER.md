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
| P-04 | The graph memory is better than plain vector retrieval (implicit in §2.2, §6) | refuted on synthetic (R-2026-10-20-01); pending E3 | Synthetic: RAG 113/120 > Sandwich 94/120; retrieval recall 120/120 for all, so the gap is the prompt, not the graph. |
| P-05 | W = S·T·(1+E)·R carries useful structure (Eq. 1) | refuted on synthetic (R-2026-10-20-01); pending E3 | Shuffled-W Sandwich 95/120 vs 94/120. |
| P-06 | Semantic similarity S is a useful association term (Eq. 2) | untested | E3 ablation −S. |
| P-07 | Temporal closeness T is a useful association term (Eq. 3) | untested | E3 ablation −T. |
| P-08 | Emotion E is "the heuristic driving the routing of thought" / multiplies significance (§2.1, Eq. 1) | untested | E3 ablation −E (lexicon valence). |
| P-09 | Hebbian reinforcement R improves behaviour with use (Eq. 4) | untested | E3 cannot test (no learning ⇒ R≡1, see THEORY_V1 §4). Needs E9 (multi-session). |
| P-10 | (1+E_ij) is a valid multiplier (Eq. 1) | refuted (math, MATH_AUDIT #8) | E ∈ [−1,1] gives zero/negative weights; v1 uses 1+κ_E·arousal. |
| P-11 | Conscious thought = A* optimal path with an emotion+logic heuristic (§3.2) | mixed | Optimality: holds only for admissible H; v1 H proven admissible (tests). Usefulness for retrieval: untested (E3 −A*). |
| P-12 | Random-walk "subconscious" surfaces useful associations (Eq. 6) | untested | v1 implements it as personalised PageRank; E3 v1 vs cosine. |
| P-13 | High C(t) means mutually supportive concepts are active (Eq. 7) | refuted on synthetic (R-2026-10-20-01, exploratory) | C_norm was highest on update probes (0.142 vs 0.039–0.046), i.e. when *conflicting* facts were co-active. |
| P-14 | Φ (Eq. 8) measures depth of consciousness at a moment | not testable / refuted as a dynamic quantity | Consciousness: not testable. Eq. 8 has no time dependence (MATH_AUDIT #1); v1 replaces with Φ_dyn. |
| P-15 | Φ is a tractable analogue of IIT Φ (§2.3) | refuted (math, MATH_AUDIT #1) | Eq. 8 is a weighted degree sum; no partition/integration term. |
| P-16 | τ_ij governs useful formation of new connections (Eq. 9) | untested | E4 θ sweep / edge-density ablation. |
| P-17 | C(t) > λ ⇒ self-awareness (Eq. 10) | not testable | No observable for self-awareness; v1 keeps λ only as a refine trigger. |
| P-18 | C(t) < λ ⇒ primary response insufficient (§5.2) | untested (weak v0 hint) | v0 AUROC(low C_norm → wrong) 0.659, n=120, 26 wrong, no CI, not pre-registered. E2 pending. |
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
| V-04 | v1 graph retrieval recall@10 ≥ cosine on LongMemEval-S | untested | E3 |
| V-05 | Real W beats shuffled-W and random-topology controls | untested | E3 |
| V-06 | Each of S, T, E contributes (ablation lowers recall) | untested | E3 |
| V-07 | A* path adds recall over ranking by activation alone | untested | E3 −A* |
| V-08 | Low C_norm predicts wrong answers / retrieval misses (AUROC > 0.5) | untested | E2 |
| V-09 | Secondary prompt without the draft avoids anchoring on a wrong draft | supported on synthetic (R-2026-10-20-01, exploratory) | 114/120 vs 94/120; not pre-registered. |
| V-10 | Recency-weighted input helps knowledge-update questions | untested | E3 exploratory x_v1_recency |

## Counts

| Status | Count |
|---|---|
| supported | 5 (P-03, V-01, V-02, V-03, V-09) |
| refuted (empirical or math) | 7 (P-04, P-05, P-10, P-13, P-15, P-19, P-14 partly) |
| mixed | 2 (P-11, P-20) |
| untested | 14 |
| not testable | 6 (P-01, P-02, P-17, P-21, P-23, P-24) |

## Change log

| Date (IST) | Change | Run |
|---|---|---|
| 2026-10-26 | Ledger created; statuses from v0 backfill and MATH_AUDIT | R-2026-10-20-01 |
