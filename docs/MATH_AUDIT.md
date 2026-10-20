# MATH_AUDIT — "A Mathematical Framework for Artificial Consciousness" (K. Sharma)

Source: `paper.pdf` (13 pages) and `paper.txt` (`pdftotext -layout`). Equation numbers are the
paper's own (1)–(10). Equations (2), (5), (6), (8) are garbled in `paper.txt`; their form below was
checked against the rendered PDF pages (pp. 5, 6, 8).

**Ground rule.** Nothing in this framework can show that a system is conscious. Every quantity below
(W, C, Φ, λ) is an engineering quantity about a memory graph. The honest research question is
*"does this graph layer make an LLM system remember / stay consistent / answer better than simpler
alternatives?"* That question is testable. "Is it conscious / self-aware?" is not, and the paper's
claims in §1 ("testable framework for instantiating synthetic consciousness"), §4.4 ("moment of
synthetic Self-Awareness") and §7 should be removed or reworded as metaphor.

Severity: **S1** = breaks the maths or makes a claim meaningless; **S2** = under-specified, cannot be
implemented without choices the paper doesn't make; **S3** = clarity/notation.

Implementation pointers refer to `src/conslayer/`. "Test" = how the fix is checked (unit test in
`tests/test_core.py` and/or an experiment in `docs/EXPERIMENT_PLAN.md`).

---

## 1. Φ is independent of the dynamics (S1)

**Where.** §4.2, Eq. (8): `Φ = Σ_{i,j} W_ij · A_i^0 / N`, "A^0 represents the basal or resting
activation", claimed "analogous to IIT" and to "evaluate the depth of consciousness at any given moment".

**Problem.** Eq. (8) contains no A(t). Given W and a fixed A^0 it is a constant: a basal-activation-
weighted mean out-strength (with uniform A^0 it is just `A^0 · mean weighted out-degree`). It does not
change "at any given moment" unless W changes, it cannot distinguish an active from an idle system,
and it has nothing to do with IIT's Φ (which is defined over cause–effect structure and minimum-
information partitions). Adding a disconnected but heavily-weighted clique *increases* Eq. (8)
although it *decreases* integration — the opposite of what IIT's Φ intends.

**Fix (two options, both implemented in `convergence.py`).**
- Dynamic version: `Φ_dyn = (1/|𝒯|) Σ_{t∈𝒯} C_norm(t)` (time-averaged normalised co-activation, see §4).
- Integration-flavoured version: `Φ_mc(W) = min_{S⊂N} cut(S, N∖S) / min(|S|, |N∖S|)` on the
  symmetrised graph (normalised min-cut, Stoer–Wagner). It is 0 iff the graph splits into independent
  parts. Still *not* IIT Φ and *not* a consciousness measure; rename it e.g. "integration score".

**Test.** `test_phi_paper_independent_of_dynamics` (Eq. 8 unchanged under any activation trajectory),
`test_phi_dynamic_depends_on_activation`, `test_phi_mincut_zero_for_disconnected`. Experimentally:
report whether Φ_dyn / Φ_mc correlate with task accuracy at all (EXP-7). If not, drop Φ.

## 2. No activation update rule (S1)

**Where.** §4.1: "Let A_i(t) represent the activation level of node i at time t." Eq. (7) and (10)
depend on A(t); no equation says how A(t) evolves, how it is initialised from an input, or its range.

**Problem.** Without A(t+1)=f(A(t),W,input), C(t) and the λ trigger are not computable, and "phase
transition" (§4.4) has no dynamics to transition in. Any convergence claim is vacuous.

**Fix.** Spreading activation with decay and input:
`A(t+1) = ζ · Pᵀ A(t) + (1−ζ) · I(t)`, `P_ij = W_ij / Σ_k W_ik` (Eq. 6), `ζ ∈ [0,1)`,
`I(t) ≥ 0, ‖I‖₁ = 1` = cue similarity of the current query/answer to each node.
Because P is row-stochastic, `‖ζPᵀ(a−b)‖₁ ≤ ζ‖a−b‖₁`: a contraction with a unique fixed point
(= personalised PageRank of I). This links §3.3 (random walk) and §4.1 (activation) into one model.
(`convergence.update_activation`)

**Test.** `test_activation_update_contracts_to_fixed_point`; used in the prototype retrieval.

## 3. λ direction contradiction (S1)

**Where.** §4.4, Eq. (10): `C(t) > λ ⟹ Self-Awareness (Self-Reflection Phase Initiated)`.
§5.2: "Analyzes the C(t) of the response. If C(t) < λ, it determines the response is insufficient.
Initiates recursive, internal self-questioning."

**Problem.** The same trigger (self-reflection / self-questioning) fires on C>λ in §4.4 and on C<λ in
§5.2. They cannot both hold. In addition λ is never given a value, a scale, or a calibration
procedure, and C is unbounded (item 4), so a fixed λ means different things for different graph sizes.

**Fix.** Separate the two roles and use the normalised C (item 4):
- *Refine trigger* (engineering, §5.2): `refine ⟺ C_norm(t) < λ` — low memory support for the draft
  ⇒ ask the secondary LLM.
- Drop the "self-awareness" reading of Eq. (10), or, if kept, make it a *separate* named state with
  its own threshold `λ_hi > λ` and an operational definition (e.g. "the system emits a report about
  its own activation state"), tested only as behaviour.
- Calibrate λ on a held-out dev set (e.g. median or a quantile of C_norm), never on the test set.
(`convergence.should_refine(c, lam, direction)` supports `below|above|always|never`.)

**Test.** `test_trigger_directions`; EXP-5 compares `below`, `above`, `always`, `never`. If `below` is
not better than `always` at lower cost, the trigger has no value.

## 4. C(t) is unbounded and scale-dependent (S1)

**Where.** Eq. (7): `C(t) = Σ_i Σ_j A_i(t) W_ij(t) A_j(t)`.

**Problem.** C = AᵀWA grows with N, with the overall scale of W (which grows without bound via R,
item 6), and with the scale of A. "High C" is undefined in absolute terms; λ (item 3) cannot be fixed.

**Fix.** `C_norm = AᵀWA / (W_max · (Σ_i A_i)²) ∈ [0,1]` for A, W ≥ 0 (since AᵀWA ≤ W_max (ΣA)²).
Invariant to rescaling A or W. (`convergence.convergence_normalized`)

**Test.** `test_convergence_normalized_in_unit_interval_and_scale_free`, `test_convergence_matches_double_sum`.

## 5. H (heuristic) undefined; A* admissibility not guaranteed (S1)

**Where.** §3.2, Eq. (5): `F(i) = g(i,j) + H(j)`; "H(j) is uniquely defined as a blend of logical
validity and emotional satisfaction relative to the goal state." No formula is given.

**Problems.**
1. "Uniquely defined" but no definition; "logical validity" of a *node* is not defined either.
2. A* returns optimal paths only if H is admissible (never overestimates cost-to-go) and, for
   graph search without re-opening, consistent. An "emotional satisfaction" term has no reason to be a
   lower bound on `Σ 1/(W+ε)`. With an inadmissible H, "conscious thought = optimal path" is false;
   A* becomes a greedy best-first heuristic search.
3. Notation: F is indexed by i but the right side by j; g(i,j) is written as a single edge cost
   whereas A* needs the *accumulated* cost g(start→j). Should be `F(j) = g(s→j) + H(j)`.
4. A goal state is required but nothing says where it comes from in the Sandwich loop (a query has no
   known "goal node").

**Fix.** Define `b(j) = μ·(1−cos(v_j, v_goal))/2 + (1−μ)·(1−e_j)/2 ∈ [0,1]` (e_j = node valence) and
`H(j) = c_min · b(j)` for j≠goal, `H(goal)=0`, where `c_min = min_edges 1/(W+ε)`. Any path from j≠goal
has ≥1 edge of cost ≥ c_min, so H ≤ h* (admissible) and `H(i) ≤ c(i,j) + H(j)` (consistent).
The price: the emotional term can only break ties among near-equal-cost paths; it cannot steer
search strongly *and* keep optimality. That trade-off should be stated in the paper.
In the prototype: start = node most similar to the query; goal = node with highest activation after
spreading (item 2). (`astar.astar_path(heuristic="blend_scaled"|"zero"|"blend_raw")`)

**Test.** `test_astar_scaled_heuristic_is_optimal` (cost equals Dijkstra on 5 random graphs),
`test_astar_scaled_heuristic_admissible_everywhere`. EXP-4 ablates A* vs no-A* retrieval.

## 6. R_ij grows without bound (S1)

**Where.** §3.1.4, Eq. (4): `R_ij(t) = R_ij(t−1) + η · success_signal`.

**Problems.** (a) Monotone, no decay, no ceiling: frequently used edges dominate W, the random walk
(Eq. 6) collapses onto a few edges, C (Eq. 7) and Φ (Eq. 8) drift upward with usage alone. (b) Initial
value R_ij(0) is not given; if 0, every new edge has W=0 (Eq. 1 is a product). (c) If success_signal
can be negative, R and hence W can become negative (breaks Eqs. 5, 6). (d) Not actually Hebbian:
Hebb depends on co-activation `A_i A_j`, not on a task success signal (this is closer to reward-
modulated / three-factor learning).

**Fix.** `R_ij ← clip(R_ij + η·s·(R_max − R_ij) − δ·(R_ij − R_0), R_0, R_max)`, `R_0 = 1`, `s ∈ [0,1]`
(saturating, with optional decay δ toward baseline). For a genuinely Hebbian variant use
`s = A_i A_j · success`. (`graph.WeightConfig(r_mode="bounded", r_init, r_max, eta, r_decay)`)

**Test.** `test_paper_R_unbounded_vs_bounded` (paper: R = 1+200·0.5 = 101 after 200 updates; fixed:
≤ R_max). EXP-6 runs ≥500 queries and tracks the Gini of edge weights and accuracy over time.

## 7. "success" is undefined, and who measures it is unclear (S1)

**Where.** Eq. (4) `success_signal`; §3.1.4 "each time the edge (i,j) is successfully traversed".

**Problem.** No definition, range or source. At inference time there is no ground truth. If success
comes from the secondary LLM judging itself, R learns the model's own biases (self-reinforcing
confirmation loop). If it comes from labels, it leaks test information when the same items are used
for evaluation.

**Fix.** Make the source explicit and pick one per experiment:
(i) *label-free proxy*: `s = 1[max_{m∈retrieved} cos(answer, m) > θ_s]` (memory was used) — what the
prototype uses; (ii) *user feedback*: thumbs up/down; (iii) *oracle* (only on a training split,
never on the evaluation split). Report which one.

**Test.** EXP-6 compares proxy vs oracle-on-train; check whether R updates help on a *disjoint* test split.

## 8. E_ij ∈ [−1,1] makes (1+E) zero or negative (S1)

**Where.** §3.1.3: "E_ij is a scalar derived from sentiment analysis and emotional valence. The term
(1+E_ij) ensures that highly emotional connections multiply the base weight, while emotionally neutral
connections (E_ij≈0) default to a multiplier of 1."

**Problems.** Valence is conventionally in [−1,1]. Then (1+E) ∈ [0,2]: strongly *negative* edges get
weight ≈ 0 (E=−1 ⇒ W=0 exactly), so A* cost `1/(W+ε) = 1/ε` (effectively deleted), and the random walk
never visits them. If E < −1 is possible (unnormalised sentiment scores), W < 0, Eq. (6) yields
negative "probabilities" and A* costs are negative (Dijkstra/A* invalid). This also contradicts the
text: "highly emotional" negative memories are *suppressed*, not amplified — the opposite of the
psychology the paper cites (negative/arousing events are remembered better). Finally, E is pairwise
but sentiment analysis produces a per-node score; the pairwise rule is missing.

**Fix.** Separate valence e_i ∈ [−1,1] from arousal |e_i|. Either
(A) `E_ij = (|e_i| + |e_j|)/2 ∈ [0,1]`, factor `(1+E_ij) ∈ [1,2]` (emotion only amplifies), or
(B) `factor = exp(κ·(e_i+e_j)/2) ∈ [e^{−κ}, e^{κ}]` (always > 0, valence-signed).
Both keep W>0. Default in code: (A). (`WeightConfig.emotion_mode = "paper"|"arousal"|"exp"`)

**Test.** `test_paper_emotion_can_zero_the_weight` (paper form gives W=0 at E=−1),
`test_fixed_emotion_is_strictly_positive`. EXP-4 ablates E (on/off, A vs B).

## 9. B_ij (emotion bias) is undefined (S2)

**Where.** §4.3, Eq. (9): "B_ij introduces an explicit Emotion Bias (a pre-existing systemic mood)".

**Problem.** A "systemic mood" is a global state, yet B is indexed by edge (i,j). Range, source and
update rule are missing. `E_ij · B_ij` also double-counts emotion with the (1+E) in W.

**Fix.** Global mood `b(t) ∈ [−1,1]` (e.g. EMA of recent turn valences: `b ← (1−ω)b + ω·e_turn`),
and `B_ij = b(t)` so `E_ij·B_ij` = mood-congruence (positive when the edge's valence matches mood).
(`WeightConfig.mood`; the EMA update is not yet implemented — b is a constant in v0.)

**Test.** EXP-4 sets α=0 vs α>0. If mood congruence never changes edge creation measurably, drop it.

## 10. τ_ij: units/scales of the terms; "threshold" vs "score"; AC undefined (S2)

**Where.** §4.3, Eq. (9): `τ_ij = α·E_ij·B_ij + β·S_ij + ρ·R_ij`; "New thoughts (nodes or edges) are
instantiated when an Activation Condition (AC) threshold is breached".

**Problems.** (a) Terms live on different scales: E·B ∈ [−1,1], S ∈ [0,1], R ∈ [R_0, ∞) (item 6), so
for an established graph the ρR term dominates and α, β become irrelevant. (b) R_ij exists only for an
existing edge, yet τ is used to decide whether to *form a new* edge: for a new edge R is undefined
(or R_0, a constant that shifts τ uniformly). (c) τ is called a "threshold factor", but it is a score;
the threshold it is compared against (AC) is never defined. (d) Edges and nodes are conflated:
"new thoughts (nodes or edges)".

**Fix.** `τ_ij = α·e_ij·b + β·S_ij + ρ·R̃_ij`, with `R̃ = (R−R_0)/(R_max−R_0) ∈ [0,1]`,
`α+β+ρ = 1` so τ ∈ [−α, 1]. Create edge (i,j) iff `τ_ij > θ` and keep only the top-k candidates per
node (k = `max_out_degree`). Node creation is a separate rule: every stored turn/fact becomes a node.
(`graph.MemoryGraph.tau`, `WeightConfig(alpha, beta, rho, theta, max_out_degree)`)

**Test.** `test_tau_threshold_controls_edges`. EXP-4 sweeps θ ∈ {0.1,0.2,0.3,0.4} (graph density).

## 11. Is W normalised? (S2)

**Where.** Eq. (1); nowhere states a range.

**Problem.** S ∈ [0,1], T ∈ (0,1], (1+E) ∈ [0,2], R ∈ [R_0,∞) ⇒ W ∈ [0, ∞). Consequences: A* costs
`1/(W+ε)` have no scale (ε means different things at different W scales), C and Φ are unbounded, the
random walk is fine (row normalisation) but nothing else is.

**Fix.** Report W in [0,1] via `W̃ = W / W_max`, `W_max = max(1+E)·R_max` (theoretical), or row-
normalise for the Markov chain only. Consider `cost = −log W̃` instead of `1/(W+ε)`: with W̃ ∈ (0,1],
the minimum-cost path then maximises the *product* of edge strengths (a principled path likelihood),
and no ε is needed except at W̃=0. (`WeightConfig.normalize="global"`, `astar.edge_cost(mode="neglog")`)

**Test.** `test_global_normalisation_bounds_W`. EXP-4 compares cost modes.

## 12. Directed graph vs symmetric factors (S2)

**Where.** §3.1: "directed, weighted graph". Eqs. (2) and (3) are symmetric (cos and |t_i−t_j|); (1+E)
with a pairwise mean is symmetric.

**Problem.** At creation, W_ij = W_ji, so the graph is effectively undirected; only R (via traversal
direction) can break symmetry later. Directedness is claimed but not used. Also, the paper doesn't say
whether an edge i→j implies j→i exists.

**Fix.** Either declare the graph undirected (simpler, honest), or add a real asymmetry, e.g. a
time-arrow factor `D_ij = 1 if t_j ≥ t_i else δ_back` (forward association stronger, cf. temporal
contiguity effects in human free recall), or asymmetric R updates (only the traversed direction).
(`WeightConfig.direction="symmetric"|"forward"`; R is already direction-specific.)

**Test.** `test_reinforce_breaks_symmetry_only_in_traversed_direction`, `test_forward_direction_option`.

## 13. S_ij is called a "distance" but is a similarity; notation (S3)

**Where.** §3.1.1: "S_ij represents the semantic distance…", Eq. (2) `S_ij = max(0, v_i·v_j/(‖v_i‖‖v_j‖))`.
Cosine is a similarity (high = close). Also "spatial" is a misnomer for embedding space; "semantic
similarity" is clearer. Clipping at 0 discards antonymy/contradiction information, which may be exactly
what a consistency checker needs (see item 18).

## 14. T_ij: which timestamp, which units, and γ scale (S2)

**Where.** §3.1.2: "t_i … when the respective memories were formed **or** last activated."

**Problems.** (a) "Formed" vs "last activated" give very different dynamics: with "last activated",
every retrieval changes T for all edges of that node (W becomes non-stationary; retrieval itself
rewrites memory). (b) Units of t (seconds? turns? sessions?) are unspecified, so γ is meaningless
without them. (c) T_ij is closeness *between two memories*, not recency relative to *now*; it cannot
by itself prefer the latest version of an updated fact (a key memory ability; see LongMemEval
"knowledge update").

**Fix.** Use formation time, units = turns (or seconds, stated), `γ = ln 2 / h` with an explicit
half-life h. Add a separate recency term for retrieval: `rec_i(now) = exp(−γ_r(now − t_i))`, applied to
the activation input I (not to W). (Prototype: γ = 0.05/turn ⇒ half-life ≈ 14 turns; recency is
only given to the LLM via "[turn k]" tags — v0 does not yet implement rec_i.)

## 15. Secondary LLM "updates W_ij" — over-determined (S2)

**Where.** §5.3: "it updates the internal state representations and edge weights (W_ij)".

**Problem.** W is *defined* by Eq. (1) from S, T, E, R. If the LLM writes W directly, Eq. (1) no longer
holds. Which factors may the LLM change (R? E? node text/embeddings?) is unspecified.

**Fix.** The secondary LLM may only (i) add nodes (new facts/summaries), (ii) supply success s for
Eq. (4), (iii) optionally relabel node valence. W is always recomputed from Eq. (1).

## 16. Eq. (6) edge cases: dangling nodes, zero rows, reducibility (S2)

**Where.** Eq. (6): `P(i→j) = W_ij / Σ_{k∈N} W_ik`.

**Problem.** Undefined (0/0) for nodes with no out-weight (and for every edge zeroed by item 8);
negative W breaks it; a reducible graph has no unique stationary distribution, so "surfacing ideas
based on accumulated weights" depends on the start node in an unspecified way.

**Fix.** Self-loop (or uniform) for dangling rows; clip W at 0; add restart probability r:
`P' = (1−r)P + r·1sᵀ` ⇒ ergodic, unique stationary distribution (personalised PageRank).
(`random_walk.transition_matrix`, `personalized_stationary`)

**Test.** `test_transition_rows_sum_to_one_and_dangling`, `test_negative_weights_are_clipped`,
`test_random_walk_empirical_matches_P`.

## 17. Scalability (S3)

Eq. (1)–(3) are pairwise; computing all W_ij is O(N²) per update; the paper doesn't bound degree.
Fix: candidate edges only among top-k nearest neighbours (ANN index), as done via `max_out_degree`.

## 18. Missing: how does the graph detect *inconsistency*? (S2)

The paper promises reflective, consistent behaviour, but no term represents contradiction. Two
contradictory facts ("I live in Kochi" / "I now live in Leeds") have *high* S and so a *strong*
positive edge — the graph treats contradiction as support, and C rises. Fix: add a signed
"conflict" relation (e.g. an NLI model's contradiction probability, or same-attribute-different-value
detection), store it as a separate edge type, and have it *reduce* C or force refinement.
Test: knowledge-update and contradiction probes (EXP-1/2).

## 19. Claims that are not falsifiable as stated (S1, scientific)

- "consciousness … is a specific mathematical and structural process" (§1) — definitional; no
  observation could refute it.
- "C(t) > λ ⟹ Self-Awareness" (§4.4) — a scalar crossing a threshold is a trigger, not awareness.
- "Φ … analogous to IIT" (§2.3, §4.2) — see item 1; shares only the symbol.
- "testable framework" (§1) — the paper states no prediction. Replace with concrete hypotheses
  (see EXPERIMENT_PLAN.md: H1–H4), all about task behaviour.
- "EEG/BCI integration … real-time emotional state control" (§6) — out of scope; remove or mark speculative.

## 20. Related work missing (S2, positioning)

The weighting is close to existing LLM memory work, which reviewers will ask about:
Generative Agents' retrieval score = recency + importance + relevance (Park et al. 2023,
arXiv:2304.03442) ≈ T, E, S here; HippoRAG uses personalised PageRank over a memory graph
(Gutiérrez et al. 2024, arXiv:2405.14831) ≈ Eq. (6) + activation; Self-Refine (Madaan et al. 2023,
arXiv:2303.17651) ≈ the secondary-LLM loop; MemGPT (Packer et al. 2023, arXiv:2310.08560) for tiered
memory. The paper's novelty, if any, must be argued against these.

---

## Summary table

| # | Item | Sev. | Fix (short) | Code option | Unit test |
|---|------|------|-------------|-------------|-----------|
| 1 | Φ static | S1 | Φ_dyn = mean C_norm(t); or normalised min-cut | `phi_dynamic`, `phi_mincut` | yes |
| 2 | no A(t) rule | S1 | A ← ζPᵀA + (1−ζ)I (contraction) | `update_activation` | yes |
| 3 | λ direction | S1 | refine iff C_norm<λ; λ from dev set | `should_refine` | yes |
| 4 | C unbounded | S1 | C_norm ∈ [0,1] | `convergence_normalized` | yes |
| 5 | H undefined/inadmissible | S1 | H = c_min·blend ∈ [0,c_min] | `astar_path(heuristic=)` | yes |
| 6 | R unbounded | S1 | saturating, clipped, decay | `r_mode="bounded"` | yes |
| 7 | success undefined | S1 | state source: proxy / feedback / oracle-on-train | `sandwich()` proxy | exp |
| 8 | (1+E) ≤ 0 | S1 | arousal (1+|e|) or exp(κe) | `emotion_mode` | yes |
| 9 | B_ij undefined | S2 | global mood b(t), B_ij=b | `mood` (const in v0) | exp |
| 10 | τ scales / AC | S2 | normalise terms, Σ weights=1, θ explicit | `tau`, `theta` | yes |
| 11 | W not normalised | S2 | W/W_max; cost = −log W̃ | `normalize`, `edge_cost` | yes |
| 12 | directed vs symmetric | S2 | undirected, or time-arrow factor | `direction` | yes |
| 13 | "distance" vs similarity | S3 | rename | — | — |
| 14 | T timestamp/units | S2 | formation time, turns, half-life; separate recency | partial | — |
| 15 | LLM writes W | S2 | LLM writes nodes/s only | design | — |
| 16 | Eq. 6 edge cases | S2 | dangling, clip, restart | `transition_matrix` | yes |
| 17 | O(N²) | S3 | top-k candidates | `max_out_degree` | yes |
| 18 | no contradiction term | S2 | signed conflict edges | not yet | exp |
| 19 | unfalsifiable claims | S1 | restate as task hypotheses | — | — |
| 20 | related work | S2 | position vs Generative Agents, HippoRAG, Self-Refine, MemGPT | — | — |
