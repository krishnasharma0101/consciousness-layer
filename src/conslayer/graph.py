"""Memory graph with the paper's edge weight, Eq. (1):

    W_ij = S_ij * T_ij * (1 + E_ij) * R_ij

Every factor can be switched off (replaced by 1) for ablations. Options marked
'paper' reproduce the paper literally; the defaults are the MATH_AUDIT fixes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import networkx as nx


@dataclass
class WeightConfig:
    use_S: bool = True
    use_T: bool = True
    use_E: bool = True
    use_R: bool = True
    gamma: float = 0.05              # Eq. (3) decay constant, units: 1 / time-unit (here: turns)
    # Emotion: 'paper' -> (1+E) with E=valence in [-1,1] (can be 0 or negative weight factor)
    #          'arousal' -> (1+E) with E=mean |valence| in [0,1]  => factor in [1,2]   (audit fix A)
    #          'exp' -> exp(kappa * E_valence) in [e^-kappa, e^kappa], always > 0    (audit fix B)
    emotion_mode: str = "arousal"
    kappa: float = 0.5
    # Learning factor: 'paper' -> R += eta * s (unbounded); 'bounded' -> R += eta*s*(R_max-R) - decay
    r_mode: str = "bounded"
    r_init: float = 1.0
    eta: float = 0.1
    r_max: float = 2.0
    r_decay: float = 0.0
    # Normalisation of W: 'none' (paper) or 'global' (divide by theoretical max -> W in [0,1])
    normalize: str = "global"
    # Directedness: 'symmetric' (paper's cosine is symmetric; only R breaks symmetry)
    #               'forward' -> multiply by back_factor when t_j < t_i (time-arrow asymmetry)
    direction: str = "symmetric"
    back_factor: float = 0.5
    eps: float = 1e-6
    # New-edge rule, Eq. (9): tau = alpha*E*B + beta*S + rho*R_norm ; edge created if tau > theta
    alpha: float = 0.2
    beta: float = 0.7
    rho: float = 0.1
    theta: float = 0.2
    mood: float = 0.0               # B: global systemic mood in [-1, 1] (audit definition of B_ij)
    max_out_degree: int = 8          # keep only the strongest k candidate edges per new node


@dataclass
class Node:
    id: int
    text: str
    emb: np.ndarray
    t: float
    valence: float = 0.0
    meta: dict = field(default_factory=dict)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MemoryGraph:
    def __init__(self, cfg: WeightConfig | None = None):
        self.cfg = cfg or WeightConfig()
        self.G = nx.DiGraph()
        self.nodes: dict[int, Node] = {}
        self.R: dict[tuple[int, int], float] = {}
        self._next = 0

    # ---------- factors ----------
    def S(self, i: int, j: int) -> float:
        """Eq. (2): S_ij = max(0, cos(v_i, v_j))."""
        if not self.cfg.use_S:
            return 1.0
        return max(0.0, cosine(self.nodes[i].emb, self.nodes[j].emb))

    def T(self, i: int, j: int) -> float:
        """Eq. (3): T_ij = exp(-gamma |t_i - t_j|)."""
        if not self.cfg.use_T:
            return 1.0
        return math.exp(-self.cfg.gamma * abs(self.nodes[i].t - self.nodes[j].t))

    def E_raw(self, i: int, j: int) -> float:
        """E_ij before the (1+E) transform. The paper does not say how a pairwise E_ij is
        obtained from per-node sentiment; we use the mean (valence) or mean |valence| (arousal)."""
        vi, vj = self.nodes[i].valence, self.nodes[j].valence
        if self.cfg.emotion_mode == "arousal":
            return (abs(vi) + abs(vj)) / 2.0
        return (vi + vj) / 2.0

    def emotion_factor(self, i: int, j: int) -> float:
        if not self.cfg.use_E:
            return 1.0
        e = self.E_raw(i, j)
        if self.cfg.emotion_mode == "exp":
            return math.exp(self.cfg.kappa * e)
        return 1.0 + e  # 'paper' and 'arousal'

    def Rf(self, i: int, j: int) -> float:
        if not self.cfg.use_R:
            return 1.0
        return self.R.get((i, j), self.cfg.r_init)

    def direction_factor(self, i: int, j: int) -> float:
        if self.cfg.direction == "forward" and self.nodes[j].t < self.nodes[i].t:
            return self.cfg.back_factor
        return 1.0

    def w_max(self) -> float:
        """Theoretical maximum of the un-normalised weight (used for 'global' normalisation)."""
        c = self.cfg
        e_max = 1.0
        if c.use_E:
            e_max = math.exp(c.kappa) if c.emotion_mode == "exp" else 2.0
        r_max = 1.0
        if c.use_R:
            r_max = c.r_max if c.r_mode == "bounded" else max([c.r_init, *self.R.values()])
        return e_max * r_max

    def weight(self, i: int, j: int) -> float:
        """Eq. (1) (optionally normalised to [0,1])."""
        w = self.S(i, j) * self.T(i, j) * self.emotion_factor(i, j) * self.Rf(i, j) * self.direction_factor(i, j)
        if self.cfg.normalize == "global":
            w /= self.w_max()
        return w

    def tau(self, i: int, j: int) -> float:
        """Eq. (9) with audit fixes: every term scaled to comparable ranges.
        E in [-1,1] (signed valence), B = global mood in [-1,1], S in [0,1], R_norm in [0,1]."""
        c = self.cfg
        e = (self.nodes[i].valence + self.nodes[j].valence) / 2.0
        r = self.Rf(i, j)
        r_norm = (r - c.r_init) / (c.r_max - c.r_init) if c.r_max > c.r_init else 0.0
        s = max(0.0, cosine(self.nodes[i].emb, self.nodes[j].emb))
        return c.alpha * e * c.mood + c.beta * s + c.rho * r_norm

    # ---------- construction ----------
    def add_node(self, text: str, emb: np.ndarray, t: float, valence: float = 0.0, **meta) -> int:
        nid = self._next
        self._next += 1
        self.nodes[nid] = Node(nid, text, np.asarray(emb, dtype=np.float32), float(t), float(valence), meta)
        self.G.add_node(nid)
        cands = []
        for j in self.nodes:
            if j == nid:
                continue
            if self.tau(nid, j) > self.cfg.theta:
                cands.append((self.weight(nid, j) + self.weight(j, nid), j))
        cands.sort(reverse=True)
        for _, j in cands[: self.cfg.max_out_degree]:
            self.G.add_edge(nid, j)
            self.G.add_edge(j, nid)
        self.refresh_weights()
        return nid

    def refresh_weights(self) -> None:
        for i, j in self.G.edges:
            self.G[i][j]["W"] = self.weight(i, j)

    # ---------- learning ----------
    def reinforce(self, path: list[int], success: float) -> None:
        """Eq. (4). success in [0,1] (paper leaves it undefined; see MATH_AUDIT)."""
        c = self.cfg
        for i, j in zip(path[:-1], path[1:]):
            r = self.R.get((i, j), c.r_init)
            if c.r_mode == "paper":
                r = r + c.eta * success
            else:
                r = r + c.eta * success * (c.r_max - r) - c.r_decay * (r - c.r_init)
                r = min(max(r, c.r_init), c.r_max)
            self.R[(i, j)] = r
        self.refresh_weights()

    # ---------- controls ----------
    def shuffle_weights(self, seed: int = 0) -> None:
        """Control: keep topology, randomly permute W across edges."""
        rng = np.random.default_rng(seed)
        edges = list(self.G.edges)
        ws = np.array([self.G[i][j]["W"] for i, j in edges])
        rng.shuffle(ws)
        for (i, j), w in zip(edges, ws):
            self.G[i][j]["W"] = float(w)

    def adjacency(self, order: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
        order = order if order is not None else sorted(self.nodes)
        idx = {n: k for k, n in enumerate(order)}
        A = np.zeros((len(order), len(order)))
        for i, j, d in self.G.edges(data=True):
            if i in idx and j in idx:
                A[idx[i], idx[j]] = d.get("W", 0.0)
        return A, order
