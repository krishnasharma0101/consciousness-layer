# THEORY v1 — a fully specified Consciousness Layer (memory-graph) model

Version history is at the bottom; **current version: v1.0** (2026-10-26). Code: `src/conslayer/theory_v1.py`
(`V1Config`, `V1Graph`), tests: `tests/test_v1.py`.

**Scope.** v1 is a specification of a *graph memory layer between two LLM calls*. It makes no claim about
consciousness. "C", "Φ", "λ" are engineering quantities about activation on a memory graph. Every design choice
carries a provenance tag:

- **[M]** mathematically required (otherwise a quantity is undefined, unbounded, or an algorithm is not correct);
- **[E:run-id]** empirically motivated by a logged run;
- **[A]** arbitrary choice (a default someone had to pick; should be swept or justified later);
- **[P]** kept from the paper as a hypothesis to be tested.

Audit items refer to `docs/MATH_AUDIT.md` (#1–#20).

## 1. Objects

- Memory node i: text x_i, unit embedding v_i ∈ R^d (MiniLM-L6-v2, d=384) [A], formation time t_i
  (units stated per dataset: turns for synthetic, days for LongMemEval) [M, audit 14], valence e_i ∈ [−1,1]
  (lexicon `conslayer.sentiment`) [A — the paper does not specify the sentiment source, audit 8].
- Node creation: every stored turn becomes a node [M, audit 10(d)]. Nodes are never created by τ.
- Global mood b(t) ∈ [−1,1]: EMA of turn valence, b ← (1−ω)b + ωe_turn, ω=0.1 [audit 9; A]. v1.0 default b ≡ 0
  (mood disabled) [A] — untested.
- Cue c: query embedding (for retrieval). In the Sandwich loop the cue is the user query, optionally
  concatenated with the Primary draft [P]; E3 uses the query only.

## 2. Quantities (all bounded)

| Quantity | Definition | Range | Provenance |
|---|---|---|---|
| S_ij | max(0, cos(v_i, v_j)) | [0,1] | paper Eq. 2 [P]; renamed "semantic similarity" [audit 13] |
| T_ij | exp(−ln2 · \|t_i − t_j\| / h), half-life h | (0,1] | Eq. 3 [P]; formation time, explicit units and half-life [M, audit 14]; h = 13.86 turns (= v0 γ 0.05/turn) or 7 days on LongMemEval [A] |
| F_ij (emotion factor) | 1 + κ_E · (\|e_i\| + \|e_j\|)/2, κ_E = 1 | [1, 2] | replaces (1+E_ij) which can be 0 or <0 [M, audit 8]; arousal form (option A) [A] |
| R_ij | R_0 = 1; on traversal i→j with success s∈[0,1]: R ← clip(R + η s (R_max − R) − δ(R − R_0), R_0, R_max); η=0.1, R_max=2, δ=0 | [1, 2] | Eq. 4 [P] made bounded/saturating [M, audit 6]; directional (only traversed direction) [audit 12]; η, R_max [A] |
| W_ij | S·T·F·R / W_max, W_max = (1+κ_E)·R_max, on edges only, else 0; any ablated factor ≡ 1 and is dropped from W_max | [0,1] | Eq. 1 [P] + normalisation [M, audit 11] |
| τ_ij | α e_ij b + β S_ij + ρ R̃_ij, e_ij = (e_i+e_j)/2, R̃ = (R−R_0)/(R_max−R_0); α=0.2, β=0.7, ρ=0.1 (sum 1) | [−α, 1] | Eq. 9 [P], comparable scales, explicit threshold [M, audit 10]; weights [A] |
| Edge set 𝓔 | for every i keep the top-k_e (k_e = 8) j by τ_ij with τ_ij > θ (θ = 0.2), then symmetrise (union) | — | AC threshold made explicit [M, audit 10]; top-k for O(N k) [audit 17]; undirected topology [audit 12]; order-independent batch build [A] (v0 built incrementally) |
| P | P_ij = W_ij / Σ_k W_ik; rows with no weight get a self-loop | row-stochastic | Eq. 6 [P] + dangling fix [M, audit 16] |
| I(c) | s_i = max(0, cos(c, v_i)); keep the top m = 5 nodes; I = s / ‖s‖₁ (uniform if all 0). Optional recency: s_i · exp(−ln2 (t_now − t_i)/h_r), h_r = 30 days (off by default) | simplex | input missing in paper [M, audit 2]; m [A]; recency [audit 14, A] |
| A(t) | A(0) = I; A(t+1) = ζ Pᵀ A(t) + (1−ζ) I, ζ = 0.85, until ‖ΔA‖₁ < 1e−8 (≤100 iters) | simplex; unique fixed point A* = personalised PageRank of I | missing in paper [M, audit 2]; ζ = v0 value [A] |
| C_norm | Aᵀ W A / (Σ_i A_i)² | [0,1] (since W ∈ [0,1]) | Eq. 7 [P] normalised [M, audit 4]. NB: v0 divided by the *observed* max W; v1 by the theoretical max (=1). |
| Φ_dyn | mean over the activation trajectory of C_norm(A(t)) | [0,1] | replaces static Eq. 8 [M, audit 1]. Not IIT Φ. |
| Φ_mc (optional) | normalised min-cut of the symmetrised W | ≥0 | integration-flavoured structural score [audit 1]; `convergence.phi_mincut` |
| A* edge cost | c_ij = −log W_ij (W clipped to [ε,1]) | [0, −log ε] | Eq. 5 used 1/(W+ε); −log makes path cost = −log Π W (path likelihood) [audit 11; A]; `cost_mode="inverse"` available |
| H(j) | c_min · [μ(1−cos(v_j,v_goal))/2 + (1−μ)(1−e_j)/2], H(goal)=0, μ = 0.7, c_min = min edge cost | [0, c_min] | paper's undefined "blend of logic and emotion"; scaled by c_min ⇒ admissible and consistent [M, audit 5]; μ [A]. Price: emotion can only break near-ties. |
| start, goal | start = argmax I (best cue match); goal = argmax A* | — | paper gives no goal [M, audit 5.4]; choice [A] |
| success s | label-free proxy: s = 1[max_{m ∈ retrieved} cos(answer, v_m) > 0.5] | {0,1} | undefined in paper [M, audit 7]; proxy (option i) [A]; oracle only on train splits |
| λ trigger | refine ⇔ C_norm < λ (direction "below", §5.2), λ = a quantile of C_norm on a dev split | — | contradiction §4.4 vs §5.2 resolved [M, audit 3]; direction [P] — **v0 evidence against it** [E:R-2026-10-20-01], see §4 |

## 3. Algorithm (one user turn)

```
ingest(turn x, time t):
    v ← embed(x); e ← valence(x); add node (x, v, t, e)
    rebuild/extend edges: top-k_e by τ with τ > θ, symmetrise; recompute W, P      # §2

answer(query q):
    draft ← PrimaryLLM(recent window, q)                                            # System 1 [P]
    cue ← embed(q)            (v1.0; v0 used q + draft)
    I ← input(cue); A ← iterate A ← ζ PᵀA + (1−ζ) I to fixed point                  # spreading activation
    C ← C_norm(A); Φ_dyn ← mean_t C_norm(A(t))
    goal ← argmax A; start ← argmax I; path ← A*(start → goal; cost −log W, H as §2)
    M ← dedupe([goal] + path + nodes sorted by A) truncated to k                    # retrieved memories
    if should_refine(C, λ):                                                         # §5.2 [P]
        answer ← SecondaryLLM(recent window, M, q)    # NO draft in the prompt [E:R-2026-10-20-01]
    else:
        answer ← draft
    s ← success_proxy(embed(answer), {v_m : m ∈ M})
    for (i→j) on path: R_ij ← saturating update with s; recompute W, P             # Eq. 4 (bounded)
    ingest(answer?)  — v1.0: assistant answers are NOT stored as nodes [A]
    return answer
```
The Secondary LLM never writes W directly; it only produces the answer that defines s (and, optionally, new
nodes) [M, audit 15].

## 4. Parameters and defaults

| Param | Default | Status |
|---|---|---|
| embedder | all-MiniLM-L6-v2, max 256 word-pieces | [A] |
| h (T half-life) | 13.86 turns / 7 days | [A] |
| κ_E | 1.0 | [A] |
| η, R_max, δ | 0.1, 2.0, 0 | [A] |
| α, β, ρ, θ, k_e | 0.2, 0.7, 0.1, 0.2, 8 | [A] (with b=0, R̃=0: edge iff S > θ/β = 0.286, top-8) |
| m (seed nodes), ζ | 5, 0.85 | [A] (v0 values) |
| μ, cost | 0.7, −log W | [A] |
| λ, direction | dev quantile, "below" | [P], contested [E:R-2026-10-20-01] |
| Secondary prompt | memories + query, no draft | [E:R-2026-10-20-01] (114/120 vs 94/120 with draft; exploratory) |
| conflict edges (audit 18) | not implemented in v1.0 | open |

Known structural facts (follow from the definitions, verified by unit tests):
- Without learning (R ≡ R_0) dropping R only rescales W, so P, A and the ranking are unchanged
  (`test_drop_R_without_learning_does_not_change_P`). Any experiment without R updates cannot test R.
- With b = 0 and R ≡ R_0, τ depends only on S, so the topology is a thresholded cosine kNN graph; the
  T, E factors only re-weight edges that S already created.
- Ranking by the fixed point A* means retrieval = personalised PageRank seeded by the top-m cosine nodes
  (HippoRAG-like, arXiv:2405.14831, but over turns instead of entities).

## 5. Dropped or re-labelled paper claims

- "C(t) > λ ⇒ self-awareness" (Eq. 10): dropped; λ is only a refine trigger [audit 3, 19].
- Φ (Eq. 8) as a consciousness score / IIT analogue: replaced by Φ_dyn (engineering); not IIT [audit 1].
- "Secondary LLM updates W": replaced by "Secondary LLM output defines s" [audit 15].

## 6. Version history

| Version | Date (IST) | Change | Motivated by |
|---|---|---|---|
| v1.0 | 2026-10-26 | First full specification (this document) | MATH_AUDIT #1–#17; R-2026-10-20-01 (no-draft prompt) |
