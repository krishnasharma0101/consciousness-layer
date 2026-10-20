"""Eq. (7) convergence C(t), Eq. (8) Phi, Eq. (10) lambda trigger, plus the missing
activation update rule (audit fix)."""
from __future__ import annotations

import numpy as np
import networkx as nx

from .random_walk import transition_matrix


def convergence(A: np.ndarray, W: np.ndarray) -> float:
    """Eq. (7): C = sum_ij A_i W_ij A_j = A^T W A. Unbounded; grows with N and with scale of W."""
    A = np.asarray(A, dtype=float)
    return float(A @ W @ A)


def convergence_normalized(A: np.ndarray, W: np.ndarray) -> float:
    """Audit fix: C_norm = A^T W A / (W_max * (sum A)^2) in [0,1] for A>=0, W>=0.
    Scale-free, so a single lambda is meaningful across graph sizes."""
    A = np.clip(np.asarray(A, dtype=float), 0, None)
    wmax = float(W.max()) if W.size else 0.0
    s = A.sum()
    if wmax <= 0 or s <= 0:
        return 0.0
    return float(A @ W @ A) / (wmax * s * s)


def phi_paper(W: np.ndarray, A0: np.ndarray) -> float:
    """Eq. (8) as written: Phi = sum_ij W_ij A0_i / N. Depends only on static W and basal A0,
    i.e. it is a weighted out-degree average, independent of the dynamics A(t)."""
    N = W.shape[0]
    return float((W.sum(axis=1) * A0).sum() / N) if N else 0.0


def phi_dynamic(A_traj: list[np.ndarray], W: np.ndarray) -> float:
    """Audit fix (minimal): time-averaged normalised co-activation over a trajectory."""
    if not A_traj:
        return 0.0
    return float(np.mean([convergence_normalized(a, W) for a in A_traj]))


def phi_mincut(W: np.ndarray) -> float:
    """Audit fix (integration-flavoured): normalised minimum bipartition cut of the
    symmetrised graph (Stoer-Wagner). 0 if the graph splits into independent parts.
    Still NOT IIT's Phi and NOT a measure of consciousness."""
    n = W.shape[0]
    if n < 2:
        return 0.0
    Ws = (W + W.T) / 2
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if Ws[i, j] > 0:
                G.add_edge(i, j, weight=float(Ws[i, j]))
    if not nx.is_connected(G):
        return 0.0
    cut, (a, b) = nx.stoer_wagner(G)
    return float(cut / min(len(a), len(b)))


def update_activation(A: np.ndarray, W: np.ndarray, I: np.ndarray, zeta: float = 0.85) -> np.ndarray:
    """Audit fix: spreading activation  A(t+1) = zeta * P^T A(t) + (1-zeta) * I(t),
    P = row-normalised W (Eq. 6). For zeta<1 this is an L1 contraction with a unique fixed
    point (= personalised PageRank of the input I)."""
    P = transition_matrix(W)
    return zeta * (P.T @ np.asarray(A, float)) + (1 - zeta) * np.asarray(I, float)


def should_refine(c_norm: float, lam: float, direction: str = "below") -> bool:
    """Lambda trigger. The paper contradicts itself: Eq. (10) fires when C > lambda,
    Sec. 5.2 re-queries when C < lambda. 'below' (default) = Sec. 5.2: low support => refine."""
    if direction == "below":
        return c_norm < lam
    if direction == "above":
        return c_norm > lam
    if direction == "always":
        return True
    if direction == "never":
        return False
    raise ValueError(direction)
