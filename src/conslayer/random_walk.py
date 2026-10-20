"""Eq. (6): subconscious thought as a random walk  P(i->j) = W_ij / sum_k W_ik.

Audit fixes: dangling rows (no out-weight) and reducibility are handled with an optional
restart (teleport) probability, which makes the chain ergodic and the stationary distribution
unique (this is personalised PageRank)."""
from __future__ import annotations

import numpy as np

from .graph import MemoryGraph


def transition_matrix(W: np.ndarray, dangling: str = "self") -> np.ndarray:
    W = np.clip(np.asarray(W, dtype=float), 0.0, None)   # negative W would break Eq. (6)
    n = W.shape[0]
    rows = W.sum(axis=1, keepdims=True)
    P = np.divide(W, rows, out=np.zeros_like(W), where=rows > 0)
    for i in np.where(rows.ravel() == 0)[0]:
        if dangling == "self":
            P[i, i] = 1.0
        else:
            P[i, :] = 1.0 / n
    return P


def random_walk(mg: MemoryGraph, start: int, steps: int, rng: np.random.Generator,
                restart: float = 0.0, restart_nodes: list[int] | None = None) -> list[int]:
    W, order = mg.adjacency()
    P = transition_matrix(W)
    idx = {n: k for k, n in enumerate(order)}
    cur = idx[start]
    seeds = [idx[n] for n in (restart_nodes or [start])]
    out = [start]
    for _ in range(steps):
        if restart > 0 and rng.random() < restart:
            cur = seeds[rng.integers(len(seeds))]
        else:
            cur = rng.choice(len(order), p=P[cur])
        out.append(order[cur])
    return out


def personalized_stationary(W: np.ndarray, seed_dist: np.ndarray, restart: float = 0.15,
                            tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    P = transition_matrix(W)
    s = np.asarray(seed_dist, dtype=float)
    s = s / s.sum()
    pi = s.copy()
    for _ in range(max_iter):
        new = (1 - restart) * pi @ P + restart * s
        if np.abs(new - pi).sum() < tol:
            return new
        pi = new
    return pi
