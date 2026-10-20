import math
import numpy as np
import networkx as nx
import pytest

from conslayer import (MemoryGraph, WeightConfig, astar_path, transition_matrix,
                       personalized_stationary, convergence, convergence_normalized,
                       phi_paper, phi_dynamic, phi_mincut, update_activation, should_refine)
from conslayer.graph import cosine


def make_graph(cfg=None, n=8, seed=0, valences=None):
    rng = np.random.default_rng(seed)
    mg = MemoryGraph(cfg or WeightConfig(theta=-1.0))  # theta=-1: connect all candidates
    for k in range(n):
        v = valences[k] if valences is not None else float(rng.uniform(-1, 1))
        mg.add_node(f"n{k}", rng.normal(size=16) + 2.0, t=k, valence=v)
    return mg


# ---------------- Eq. (1)-(4) ----------------
def test_weight_is_product_of_factors():
    mg = make_graph(WeightConfig(theta=-1, normalize="none", emotion_mode="arousal"))
    i, j = 0, 3
    expect = mg.S(i, j) * mg.T(i, j) * mg.emotion_factor(i, j) * mg.Rf(i, j)
    assert mg.weight(i, j) == pytest.approx(expect)


def test_S_nonnegative_and_cosine():
    mg = MemoryGraph(WeightConfig(theta=-1))
    mg.add_node("a", np.array([1.0, 0.0]), 0)
    mg.add_node("b", np.array([-1.0, 0.0]), 0)
    mg.add_node("c", np.array([1.0, 1.0]), 0)
    assert mg.S(0, 1) == 0.0
    assert mg.S(0, 2) == pytest.approx(1 / math.sqrt(2))


def test_T_decay():
    mg = make_graph(WeightConfig(theta=-1, gamma=0.5))
    assert mg.T(0, 2) == pytest.approx(math.exp(-1.0))
    assert mg.T(0, 0) == pytest.approx(1.0)


def test_paper_emotion_can_zero_the_weight():
    """Audit item: with E in [-1,1], (1+E)=0 at E=-1 -> edge vanishes; A* cost -> 1/eps."""
    cfg = WeightConfig(theta=-1, emotion_mode="paper", normalize="none")
    mg = make_graph(cfg, n=2, valences=[-1.0, -1.0])
    assert mg.emotion_factor(0, 1) == 0.0
    assert mg.weight(0, 1) == 0.0


@pytest.mark.parametrize("mode", ["arousal", "exp"])
def test_fixed_emotion_is_strictly_positive(mode):
    cfg = WeightConfig(theta=-1, emotion_mode=mode)
    mg = make_graph(cfg, n=2, valences=[-1.0, -1.0])
    assert mg.emotion_factor(0, 1) > 0


def test_ablation_toggles_replace_factor_by_one():
    cfg = WeightConfig(theta=-1, use_S=False, use_T=False, use_E=False, use_R=False, normalize="none")
    mg = make_graph(cfg)
    assert mg.weight(0, 5) == pytest.approx(1.0)


def test_paper_R_unbounded_vs_bounded():
    paper = make_graph(WeightConfig(theta=-1, r_mode="paper", eta=0.5))
    fixed = make_graph(WeightConfig(theta=-1, r_mode="bounded", eta=0.5, r_max=2.0))
    for _ in range(200):
        paper.reinforce([0, 1], 1.0)
        fixed.reinforce([0, 1], 1.0)
    assert paper.R[(0, 1)] == pytest.approx(1 + 200 * 0.5)
    assert fixed.R[(0, 1)] <= 2.0
    assert fixed.R[(0, 1)] > 1.9


def test_global_normalisation_bounds_W():
    mg = make_graph(WeightConfig(theta=-1, normalize="global"))
    for _ in range(50):
        mg.reinforce([0, 1, 2], 1.0)
    W, _ = mg.adjacency()
    assert W.min() >= 0 and W.max() <= 1.0 + 1e-9


def test_reinforce_breaks_symmetry_only_in_traversed_direction():
    mg = make_graph(WeightConfig(theta=-1))
    assert mg.weight(0, 1) == pytest.approx(mg.weight(1, 0))   # cosine & |dt| symmetric
    mg.reinforce([0, 1], 1.0)
    assert mg.weight(0, 1) > mg.weight(1, 0)


def test_forward_direction_option():
    mg = make_graph(WeightConfig(theta=-1, direction="forward", back_factor=0.5))
    assert mg.weight(3, 1) == pytest.approx(0.5 * mg.weight(1, 3))


def test_tau_threshold_controls_edges():
    sparse = make_graph(WeightConfig(theta=10.0))
    assert sparse.G.number_of_edges() == 0
    dense = make_graph(WeightConfig(theta=-1.0, max_out_degree=100))
    assert dense.G.number_of_edges() == 8 * 7


def test_shuffle_preserves_weight_multiset():
    mg = make_graph()
    before = sorted(d["W"] for *_, d in mg.G.edges(data=True))
    mg.shuffle_weights(seed=1)
    after = sorted(d["W"] for *_, d in mg.G.edges(data=True))
    assert before == pytest.approx(after)


# ---------------- Eq. (5) A* ----------------
def dijkstra_cost(mg, s, t, eps=1e-6):
    H = nx.DiGraph()
    for i, j, d in mg.G.edges(data=True):
        H.add_edge(i, j, c=1.0 / (d["W"] + eps))
    return nx.shortest_path_length(H, s, t, weight="c")


@pytest.mark.parametrize("seed", range(5))
def test_astar_scaled_heuristic_is_optimal(seed):
    mg = make_graph(WeightConfig(theta=0.2, max_out_degree=3), n=15, seed=seed)
    path, cost, _ = astar_path(mg, 0, 14, heuristic="blend_scaled")
    if path:
        assert cost == pytest.approx(dijkstra_cost(mg, 0, 14))
        assert path[0] == 0 and path[-1] == 14


def test_astar_scaled_heuristic_admissible_everywhere():
    mg = make_graph(WeightConfig(theta=0.2, max_out_degree=3), n=12, seed=3)
    from conslayer.astar import blend, edge_cost
    c_min = min(edge_cost(d["W"]) for *_, d in mg.G.edges(data=True))
    goal = 11
    H = nx.DiGraph([(i, j, {"c": edge_cost(d["W"])}) for i, j, d in mg.G.edges(data=True)])
    for j in mg.G.nodes:
        if j == goal or not nx.has_path(H, j, goal):
            continue
        assert c_min * blend(mg, j, goal, 0.7) <= nx.shortest_path_length(H, j, goal, weight="c") + 1e-9


def test_astar_unreachable_returns_empty():
    mg = make_graph(WeightConfig(theta=10.0), n=3)
    path, cost, _ = astar_path(mg, 0, 2)
    assert path == [] and math.isinf(cost)


# ---------------- Eq. (6) random walk ----------------
def test_transition_rows_sum_to_one_and_dangling():
    W = np.array([[0, 2, 2], [1, 0, 0], [0, 0, 0]], float)
    P = transition_matrix(W)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[2, 2] == 1.0
    assert P[0, 1] == pytest.approx(0.5)


def test_negative_weights_are_clipped():
    W = np.array([[0, -1, 1], [1, 0, 1], [1, 1, 0]], float)
    P = transition_matrix(W)
    assert P.min() >= 0 and np.allclose(P.sum(axis=1), 1.0)


def test_random_walk_empirical_matches_P():
    from conslayer import random_walk
    mg = make_graph(n=5, seed=2)
    rng = np.random.default_rng(0)
    walk = random_walk(mg, 0, 20000, rng)
    W, order = mg.adjacency()
    pi = personalized_stationary(W, np.ones(len(order)), restart=1e-9, max_iter=20000)
    emp = np.bincount([order.index(n) for n in walk], minlength=len(order)) / len(walk)
    assert np.abs(emp - pi).max() < 0.03


# ---------------- Eq. (7), (8), (10) ----------------
def test_convergence_matches_double_sum():
    rng = np.random.default_rng(0)
    W = rng.random((6, 6)); A = rng.random(6)
    assert convergence(A, W) == pytest.approx(sum(A[i] * W[i, j] * A[j] for i in range(6) for j in range(6)))


def test_convergence_normalized_in_unit_interval_and_scale_free():
    rng = np.random.default_rng(1)
    for _ in range(50):
        W = rng.random((7, 7)); A = rng.random(7)
        c = convergence_normalized(A, W)
        assert 0.0 <= c <= 1.0
        assert convergence_normalized(A, 5 * W) == pytest.approx(c)
        assert convergence_normalized(3 * A, W) == pytest.approx(c)


def test_phi_paper_independent_of_dynamics():
    """Audit item: Eq. (8) contains no A(t); any activation trajectory gives the same Phi."""
    rng = np.random.default_rng(0)
    W = rng.random((5, 5)); A0 = np.full(5, 0.1)
    p1 = phi_paper(W, A0)
    _ = [update_activation(rng.random(5), W, rng.random(5)) for _ in range(10)]
    assert phi_paper(W, A0) == p1


def test_phi_dynamic_depends_on_activation():
    rng = np.random.default_rng(0)
    W = rng.random((5, 5))
    a = phi_dynamic([np.array([1, 0, 0, 0, 0.0])], W)
    b = phi_dynamic([np.ones(5)], W)
    assert a != b


def test_phi_mincut_zero_for_disconnected():
    W = np.zeros((4, 4)); W[0, 1] = W[1, 0] = 1; W[2, 3] = W[3, 2] = 1
    assert phi_mincut(W) == 0.0
    assert phi_mincut(np.ones((4, 4)) - np.eye(4)) > 0


def test_activation_update_contracts_to_fixed_point():
    rng = np.random.default_rng(0)
    W = rng.random((6, 6)); I = rng.random(6)
    a, b = rng.random(6), rng.random(6)
    for _ in range(200):
        a, b = update_activation(a, W, I), update_activation(b, W, I)
    assert np.allclose(a, b, atol=1e-8)


def test_trigger_directions():
    assert should_refine(0.1, 0.3, "below") and not should_refine(0.5, 0.3, "below")
    assert should_refine(0.5, 0.3, "above") and not should_refine(0.1, 0.3, "above")
    assert should_refine(0.0, 1.0, "always") and not should_refine(0.0, 1.0, "never")
