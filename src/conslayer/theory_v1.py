"""Theory v1 of the Consciousness Layer (see docs/THEORY_V1.md for definitions and provenance).

Vectorised, order-independent implementation for batch memory graphs (hundreds of nodes per
haystack). Every quantity is bounded:
    S_ij = max(0, cos(v_i, v_j))                          in [0,1]
    T_ij = exp(-ln2 * |t_i - t_j| / h)                     in (0,1]   (h = half-life, stated units)
    F_ij = 1 + kappa_E * (|e_i| + |e_j|) / 2               in [1, 1+kappa_E]   (arousal; audit 8)
    R_ij in [R0, Rmax], saturating update                  (audit 6)
    W_ij = S^uS T^uT F^uE R^uR / ((1+kappa_E)^uE Rmax^uR)  in [0,1]   on the edge set, 0 elsewhere
    tau_ij = alpha*e_ij*b + beta*S_ij + rho*Rtilde_ij      edge iff tau > theta, top-k per node
    P = row-normalised W (dangling -> self loop)           (audit 16)
    A <- zeta P^T A + (1-zeta) I                           (audit 2; fixed point = personalised PageRank)
    C_norm = A^T W A / (sum A)^2                           in [0,1]   (audit 4)
    Phi_dyn = mean_t C_norm(A_t) over the activation trajectory       (audit 1)
Nothing here measures consciousness.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, asdict, field

import numpy as np

LN2 = math.log(2.0)


@dataclass
class V1Config:
    # factor switches (ablations replace the factor by 1 and drop it from W_max)
    use_S: bool = True
    use_T: bool = True
    use_E: bool = True
    use_R: bool = True
    half_life: float = 13.86        # in `time_unit`; v0 gamma=0.05/turn  <=>  h = ln2/0.05 = 13.86 turns
    time_unit: str = "turns"
    kappa_E: float = 1.0            # emotion amplitude: factor in [1, 1+kappa_E]
    r_init: float = 1.0
    r_max: float = 2.0
    eta: float = 0.1
    r_decay: float = 0.0
    # edge creation (Eq. 9, audit 10)
    alpha: float = 0.2
    beta: float = 0.7
    rho: float = 0.1
    theta: float = 0.2
    mood: float = 0.0               # b in [-1,1]; constant unless update_mood() is used
    mood_omega: float = 0.1
    k_edges: int = 8                # max candidate edges chosen per node (union-symmetrised)
    # activation (audit 2)
    zeta: float = 0.85
    seed_k: int = 5                 # number of cue-similar nodes that receive input
    max_iter: int = 100
    tol: float = 1e-8
    use_recency: bool = False       # audit 14: optional recency multiplier on the input I
    recency_half_life: float = 30.0
    # A* (Eq. 5, audit 5/11)
    use_astar: bool = True
    cost_mode: str = "neglog"       # 'neglog' (-log W) or 'inverse' (1/(W+eps))
    mu: float = 0.7
    eps: float = 1e-6
    # trigger (Eq. 10 / sec 5.2, audit 3)
    lam: float = 0.05
    lam_direction: str = "always"   # v1.1 default (THEORY_V1 §5); "below" = paper §5.2
    # success proxy (audit 7)
    success_threshold: float = 0.5
    # control: 'none' | 'shuffle' (permute W over edges) | 'random_topology' (same #edges, random pairs)
    control: str = "none"
    control_seed: int = 0

    def to_dict(self):
        return asdict(self)


def _cos_matrix(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, np.float64)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = V / n
    return U @ U.T


class V1Graph:
    """Memory graph over N nodes with embeddings V (N x d), times t (N,), valences e (N,)."""

    def __init__(self, V, t, e, cfg: V1Config | None = None):
        self.cfg = cfg or V1Config()
        self.V = np.asarray(V, np.float64)
        n = np.linalg.norm(self.V, axis=1, keepdims=True)
        n[n == 0] = 1.0
        self.U = self.V / n
        self.t = np.asarray(t, np.float64)
        self.e = np.clip(np.asarray(e, np.float64), -1, 1)
        self.N = len(self.t)
        self.R = np.full((self.N, self.N), self.cfg.r_init)
        self.cos = self.U @ self.U.T
        self.S_full = np.clip(self.cos, 0, None)
        np.fill_diagonal(self.S_full, 0.0)
        self.mask = self._build_edges()
        self._apply_control_topology()
        self.recompute_W()

    # ---------- factors ----------
    def factors(self):
        c = self.cfg
        one = np.ones((self.N, self.N))
        S = self.S_full if c.use_S else one
        T = np.exp(-LN2 * np.abs(self.t[:, None] - self.t[None, :]) / c.half_life) if c.use_T else one
        ar = (np.abs(self.e)[:, None] + np.abs(self.e)[None, :]) / 2
        F = 1.0 + c.kappa_E * ar if c.use_E else one
        R = self.R if c.use_R else one
        return S, T, F, R

    def w_max(self) -> float:
        c = self.cfg
        return ((1 + c.kappa_E) if c.use_E else 1.0) * (c.r_max if c.use_R else 1.0)

    def tau(self) -> np.ndarray:
        c = self.cfg
        rt = (self.R - c.r_init) / (c.r_max - c.r_init) if c.r_max > c.r_init else np.zeros_like(self.R)
        eij = (self.e[:, None] + self.e[None, :]) / 2
        tau = c.alpha * eij * c.mood + c.beta * self.S_full + c.rho * rt
        np.fill_diagonal(tau, -np.inf)
        return tau

    def _build_edges(self) -> np.ndarray:
        c = self.cfg
        tau = self.tau()
        mask = np.zeros((self.N, self.N), bool)
        if self.N < 2:
            return mask
        k = min(c.k_edges, self.N - 1)
        top = np.argpartition(-tau, k - 1, axis=1)[:, :k] if k > 0 else np.zeros((self.N, 0), int)
        rows = np.repeat(np.arange(self.N), top.shape[1])
        cols = top.ravel()
        ok = tau[rows, cols] > c.theta
        mask[rows[ok], cols[ok]] = True
        mask |= mask.T                              # undirected topology (audit 12)
        np.fill_diagonal(mask, False)
        return mask

    def _apply_control_topology(self):
        if self.cfg.control != "random_topology":
            return
        rng = np.random.default_rng(self.cfg.control_seed)
        iu = np.triu_indices(self.N, 1)
        n_e = int(np.triu(self.mask, 1).sum())
        pick = rng.choice(len(iu[0]), size=n_e, replace=False) if n_e else np.array([], int)
        m = np.zeros_like(self.mask)
        m[iu[0][pick], iu[1][pick]] = True
        self.mask = m | m.T

    def recompute_W(self):
        S, T, F, R = self.factors()
        W = S * T * F * R / self.w_max()
        W = np.where(self.mask, W, 0.0)
        if self.cfg.control == "shuffle":
            rng = np.random.default_rng(self.cfg.control_seed)
            iu = np.where(np.triu(self.mask, 1))
            vals = W[iu].copy()
            rng.shuffle(vals)
            W = np.zeros_like(W)
            W[iu] = vals
            W = W + W.T
        if self.cfg.control == "random_topology":
            # keep the multiset of real-graph weights, assign to random pairs
            real = V1Graph(self.V, self.t, self.e, V1Config(**{**self.cfg.to_dict(), "control": "none"}))
            vals = real.W[np.where(np.triu(real.mask, 1))]
            rng = np.random.default_rng(self.cfg.control_seed + 1)
            iu = np.where(np.triu(self.mask, 1))
            vals = rng.permutation(vals)[: len(iu[0])]
            W = np.zeros_like(W)
            W[iu] = vals
            W = W + W.T
        self.W = W
        rows = W.sum(axis=1, keepdims=True)
        P = np.divide(W, rows, out=np.zeros_like(W), where=rows > 0)
        dang = np.where(rows.ravel() == 0)[0]
        P[dang, dang] = 1.0
        self.P = P

    # ---------- input and activation ----------
    def input_vector(self, cue_emb, t_now: float | None = None) -> np.ndarray:
        c = self.cfg
        q = np.asarray(cue_emb, np.float64)
        q = q / (np.linalg.norm(q) or 1.0)
        s = np.clip(self.U @ q, 0, None)
        if c.use_recency and t_now is not None:
            s = s * np.exp(-LN2 * np.clip(t_now - self.t, 0, None) / c.recency_half_life)
        I = np.zeros(self.N)
        k = min(c.seed_k, self.N)
        top = np.argsort(-s)[:k]
        I[top] = s[top]
        if I.sum() <= 0:
            I[:] = 1.0 / self.N
        return I / I.sum()

    def activate(self, I: np.ndarray, return_traj: bool = False):
        c = self.cfg
        A = I.copy()
        traj = [A.copy()]
        for _ in range(c.max_iter):
            new = c.zeta * (self.P.T @ A) + (1 - c.zeta) * I
            done = np.abs(new - A).sum() < c.tol
            A = new
            traj.append(A.copy())
            if done:
                break
        return (A, traj) if return_traj else A

    # ---------- quantities ----------
    def c_norm(self, A: np.ndarray) -> float:
        A = np.clip(A, 0, None)
        s = A.sum()
        return 0.0 if s <= 0 else float(A @ self.W @ A / (s * s))

    def phi_dyn(self, traj) -> float:
        return float(np.mean([self.c_norm(a) for a in traj])) if traj else 0.0

    # ---------- A* ----------
    def edge_cost(self, w: np.ndarray | float):
        c = self.cfg
        if c.cost_mode == "neglog":
            return -np.log(np.clip(w, c.eps, 1.0))
        return 1.0 / (np.clip(w, 0, None) + c.eps)

    def heuristic(self, goal: int, c_min: float) -> np.ndarray:
        mu = self.cfg.mu
        logical = (1 - self.U @ self.U[goal]) / 2
        emotional = (1 - self.e) / 2
        h = c_min * (mu * logical + (1 - mu) * emotional)
        h[goal] = 0.0
        return h

    def astar(self, start: int, goal: int):
        """Returns (path, cost, n_expanded). Admissible + consistent heuristic (H <= c_min)."""
        if start == goal:
            return [start], 0.0, 0
        ii, jj = np.where(self.mask)
        if len(ii) == 0:
            return [], math.inf, 0
        costs = self.edge_cost(self.W[ii, jj])
        c_min = float(costs.min())
        h = self.heuristic(goal, c_min)
        nbrs: dict[int, list[tuple[int, float]]] = {}
        for a, b, cst in zip(ii.tolist(), jj.tolist(), costs.tolist()):
            nbrs.setdefault(a, []).append((b, cst))
        g = {start: 0.0}
        parent = {start: None}
        openh = [(h[start], 0.0, start)]
        closed = set()
        exp = 0
        while openh:
            _, gc, u = heapq.heappop(openh)
            if u in closed:
                continue
            if u == goal:
                path = [u]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1], gc, exp
            closed.add(u)
            exp += 1
            for v, cst in nbrs.get(u, []):
                ng = gc + cst
                if ng < g.get(v, math.inf):
                    g[v] = ng
                    parent[v] = u
                    heapq.heappush(openh, (ng + h[v], ng, v))
        return [], math.inf, exp

    # ---------- retrieval (the algorithm in THEORY_V1 section 3) ----------
    def retrieve(self, cue_emb, k: int, t_now: float | None = None) -> dict:
        I = self.input_vector(cue_emb, t_now)
        A, traj = self.activate(I, return_traj=True)
        goal = int(np.argmax(A))
        start = int(np.argmax(I))
        path = []
        if self.cfg.use_astar:
            path, _, _ = self.astar(start, goal)
        order = []
        for n in ([goal] + path if self.cfg.use_astar else []) + np.argsort(-A, kind="stable").tolist():
            if n not in order:
                order.append(n)
            if len(order) >= k:
                break
        return {"ids": order, "A": A, "I": I, "path": path, "c_norm": self.c_norm(A),
                "phi_dyn": self.phi_dyn(traj), "n_iter": len(traj) - 1, "start": start, "goal": goal}

    # ---------- v1.2 candidate: graph as a re-ranking bonus on a first-stage score ----------
    def activate_from_scores(self, first: np.ndarray) -> np.ndarray:
        """Seed I from the top-m nodes of an arbitrary first-stage score (min-max scaled, >= 0)."""
        f = _minmax(np.asarray(first, np.float64))
        I = np.zeros(self.N)
        top = np.argsort(-f, kind="stable")[: min(self.cfg.seed_k, self.N)]
        I[top] = f[top] + 1e-12
        return self.activate(I / I.sum())

    def hybrid_rank(self, first: np.ndarray, w: float) -> np.ndarray:
        """score = (1-w) * minmax(first) + w * minmax(A); w = 0 reproduces the first-stage ranking."""
        f = _minmax(np.asarray(first, np.float64))
        if w == 0:
            return np.argsort(-f, kind="stable")
        A = self.activate_from_scores(first)
        return np.argsort(-((1 - w) * f + w * _minmax(A)), kind="stable")

    # ---------- learning ----------
    def reinforce(self, path, success: float):
        """Eq. (4) with the audit-6 saturating rule, directional (traversed i->j only)."""
        c = self.cfg
        s = float(np.clip(success, 0, 1))
        for i, j in zip(path[:-1], path[1:]):
            r = self.R[i, j]
            r = r + c.eta * s * (c.r_max - r) - c.r_decay * (r - c.r_init)
            self.R[i, j] = min(max(r, c.r_init), c.r_max)
        self.recompute_W()

    def should_refine(self, c_norm: float) -> bool:
        d = self.cfg.lam_direction
        if d == "below":
            return c_norm < self.cfg.lam
        if d == "above":
            return c_norm > self.cfg.lam
        if d == "always":
            return True
        if d == "never":
            return False
        raise ValueError(d)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def rrf(rank_lists, k: int = 60, n: int | None = None) -> np.ndarray:
    """Reciprocal-rank-fusion score per item (higher = better)."""
    n = n or len(rank_lists[0])
    s = np.zeros(n)
    for r in rank_lists:
        s[np.asarray(r)] += 1.0 / (k + 1 + np.arange(len(r)))
    return s


def success_proxy(answer_emb, retrieved_embs, threshold: float = 0.5) -> float:
    """Label-free proxy (audit 7, option i): 1 if the answer is close to a retrieved memory."""
    a = np.asarray(answer_emb, np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    M = np.asarray(retrieved_embs, np.float64)
    M = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)
    return float((M @ a).max() > threshold) if len(M) else 0.0


def update_mood(b: float, e_turn: float, omega: float = 0.1) -> float:
    """Global mood b(t) as EMA of turn valence (audit 9)."""
    return float(np.clip((1 - omega) * b + omega * e_turn, -1, 1))
