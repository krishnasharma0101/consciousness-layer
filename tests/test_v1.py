"""Tests for theory v1 (src/conslayer/theory_v1.py), stats and runlog."""
import json, math, os, subprocess
import numpy as np
import pytest

from conslayer.theory_v1 import V1Config, V1Graph, success_proxy, update_mood
from conslayer import stats as st
from conslayer import runlog


def _rand_graph(n=40, d=16, seed=0, **kw):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, d)) + 2.0 * rng.normal(size=(1, d))   # shared direction -> many positive cosines
    t = np.sort(rng.uniform(0, 50, n))
    e = rng.uniform(-1, 1, n)
    return V1Graph(V, t, e, V1Config(**kw))


def test_W_bounded_symmetric_nonnegative():
    g = _rand_graph()
    assert g.W.min() >= 0 and g.W.max() <= 1 + 1e-12
    assert np.allclose(g.W, g.W.T)
    assert np.all(np.diag(g.W) == 0)


def test_emotion_factor_bounded_and_positive():
    g = _rand_graph(kappa_E=1.0)
    _, _, F, _ = g.factors()
    assert F.min() >= 1.0 and F.max() <= 2.0


def test_P_row_stochastic_with_dangling():
    V = np.eye(4)                      # orthogonal -> S=0 -> no edges -> all dangling
    g = V1Graph(V, np.arange(4.0), np.zeros(4), V1Config())
    assert np.allclose(g.P.sum(axis=1), 1.0)
    assert np.allclose(np.diag(g.P), 1.0)


def test_activation_converges_to_ppr_fixed_point():
    g = _rand_graph()
    I = g.input_vector(g.V[3])
    A = g.activate(I)
    z = g.cfg.zeta
    A_closed = np.linalg.solve(np.eye(g.N) - z * g.P.T, (1 - z) * I)
    assert np.allclose(A, A_closed, atol=1e-6)
    assert abs(A.sum() - 1) < 1e-6


def test_c_norm_unit_interval_and_scale_free_in_A():
    g = _rand_graph()
    A = g.activate(g.input_vector(g.V[0]))
    c = g.c_norm(A)
    assert 0 <= c <= 1
    assert math.isclose(c, g.c_norm(7.3 * A), rel_tol=1e-9)


def test_phi_dyn_depends_on_trajectory():
    g = _rand_graph()
    _, tr1 = g.activate(g.input_vector(g.V[0]), return_traj=True)
    _, tr2 = g.activate(g.input_vector(g.V[20]), return_traj=True)
    assert g.phi_dyn(tr1) != g.phi_dyn(tr2)


def test_astar_matches_dijkstra_and_is_admissible():
    import networkx as nx
    for seed in range(5):
        g = _rand_graph(seed=seed, theta=0.0, k_edges=6)
        ii, jj = np.where(g.mask)
        G = nx.DiGraph()
        for a, b in zip(ii, jj):
            G.add_edge(int(a), int(b), c=float(g.edge_cost(g.W[a, b])))
        nodes = list(G.nodes)
        s, t = nodes[0], nodes[-1]
        if not nx.has_path(G, s, t):
            continue
        path, cost, _ = g.astar(s, t)
        assert math.isclose(cost, nx.dijkstra_path_length(G, s, t, weight="c"), rel_tol=1e-9)
        c_min = min(d["c"] for _, _, d in G.edges(data=True))
        h = g.heuristic(t, c_min)
        dist = nx.single_source_dijkstra_path_length(G.reverse(), t, weight="c")
        for n, dn in dist.items():
            assert h[n] <= dn + 1e-12


def test_R_saturates_and_is_directional():
    g = _rand_graph(theta=0.0)
    i, j = map(int, np.argwhere(g.mask)[0])
    for _ in range(500):
        g.reinforce([i, j], 1.0)
    assert g.R[i, j] <= g.cfg.r_max + 1e-12 and g.R[i, j] > 1.9
    assert g.R[j, i] == g.cfg.r_init


def test_ablation_flags_set_factor_to_one():
    g = _rand_graph(use_S=False, use_T=False, use_E=False, use_R=False)
    # W is 1 on every edge (all factors 1, W_max 1)
    assert np.allclose(g.W[g.mask], 1.0)


def test_drop_R_without_learning_does_not_change_P():
    a = _rand_graph()
    b = _rand_graph(use_R=False)
    assert np.allclose(a.P, b.P)


def test_shuffle_keeps_topology_and_weight_multiset():
    a = _rand_graph()
    b = _rand_graph(control="shuffle", control_seed=3)
    assert np.array_equal(a.mask, b.mask)
    assert np.allclose(np.sort(a.W[a.mask]), np.sort(b.W[b.mask]))
    assert not np.allclose(a.W, b.W)


def test_random_topology_same_edge_count():
    a = _rand_graph()
    b = _rand_graph(control="random_topology", control_seed=3)
    assert a.mask.sum() == b.mask.sum()
    assert not np.array_equal(a.mask, b.mask)


def test_theta_controls_density():
    dens = [_rand_graph(theta=th).mask.sum() for th in (0.0, 0.3, 0.6, 0.9)]
    assert dens == sorted(dens, reverse=True)


def test_retrieve_returns_k_unique_and_includes_goal():
    g = _rand_graph()
    r = g.retrieve(g.V[5], k=7)
    assert len(r["ids"]) == 7 and len(set(r["ids"])) == 7
    assert r["goal"] in r["ids"]


def test_recency_changes_input():
    g = _rand_graph(use_recency=True, recency_half_life=5.0, seed_k=40)
    I1 = g.input_vector(g.V[0], t_now=60.0)
    g.cfg.use_recency = False
    I0 = g.input_vector(g.V[0], t_now=60.0)
    assert not np.allclose(I0, I1)


def test_trigger_and_proxy_and_mood():
    g = _rand_graph(lam=0.1)
    assert g.should_refine(1.0)            # v1.1 default: always
    g.cfg.lam_direction = "below"
    assert g.should_refine(0.05) and not g.should_refine(0.2)
    g.cfg.lam_direction = "above"
    assert g.should_refine(0.2)
    assert success_proxy(np.ones(3), np.ones((2, 3))) == 1.0
    assert success_proxy(np.array([1, 0, 0.]), np.array([[0, 1, 0.]])) == 0.0
    assert -1 <= update_mood(0.9, 1.0) <= 1


# ---------- stats ----------
def test_wilson_known_value():
    lo, hi = st.wilson(113, 120)
    assert abs(lo - 0.884) < 2e-3 and abs(hi - 0.971) < 2e-3


def test_auroc_basic_and_ci():
    assert st.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert st.auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    a, lo, hi = st.auroc_ci(np.r_[np.zeros(30), np.ones(30)] + np.random.default_rng(0).normal(0, 1, 60),
                            np.r_[np.zeros(30), np.ones(30)], B=500)
    assert lo <= a <= hi


def test_paired_tests():
    assert st.mcnemar_exact(0, 0) == 1.0
    assert st.mcnemar_exact(1, 20) < 0.001
    d = np.r_[np.ones(30), np.zeros(10)]
    assert st.sign_flip_p(d, B=2000) < 0.01
    assert st.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


# ---------- runlog ----------
def test_runlog_writes_config_and_refuses_reuse(tmp_path):
    r = runlog.Run("R-test-01", {"a": 1}, seed=0, results_root=str(tmp_path), check_git=False)
    r.save_csv("x.csv", [{"a": 1}])
    r.finish(summary={"ok": True})
    meta = json.load(open(tmp_path / "R-test-01" / "config.json"))
    assert meta["status"] == "done" and meta["files"] == ["x.csv"] and meta["seed"] == 0
    assert "+05:30" in meta["start_ist"]
    with pytest.raises(RuntimeError):
        runlog.Run("R-test-01", {}, seed=0, results_root=str(tmp_path), check_git=False)


def test_hybrid_rank_w0_is_first_stage_and_bonus_reorders():
    from conslayer.theory_v1 import rrf
    g = _rand_graph()
    first = np.random.default_rng(3).random(g.N)
    assert list(g.hybrid_rank(first, 0.0)) == list(np.argsort(-first, kind="stable"))
    r1 = g.hybrid_rank(first, 1.0)
    A = g.activate_from_scores(first)
    assert np.isclose(A.sum(), 1.0) and r1[0] == int(np.argmax(A))
    s = rrf([np.arange(5), np.arange(5)[::-1]], k=60)
    assert np.allclose(s, s[::-1])         # symmetric fusion of reversed lists
