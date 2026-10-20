"""Eq. (5): conscious thought as A* with edge cost g = 1/(W+eps).

The paper's H(j) ('blend of logical validity and emotional satisfaction') is undefined.
Heuristics offered here:
  'zero'         : h = 0 (Dijkstra; trivially admissible)
  'blend_scaled' : h(j) = c_min * blend(j), blend in [0,1], h(goal)=0.
                   Since every remaining path from j != goal has >= 1 edge of cost >= c_min,
                   h <= true cost-to-go (admissible) and h(i) <= c(i,j) + h(j) (consistent).
  'blend_raw'    : h(j) = h_scale * blend(j) without the c_min bound -> can overestimate,
                   so A* may return a sub-optimal path (this is the paper's unspecified case).
"""
from __future__ import annotations

import heapq
import math
import numpy as np

from .graph import MemoryGraph, cosine


def edge_cost(w: float, eps: float = 1e-6, mode: str = "inverse") -> float:
    if mode == "inverse":
        return 1.0 / (max(w, 0.0) + eps)
    if mode == "neglog":  # requires W in (0,1]; additive cost = -log W  <=> multiplicative path strength
        return -math.log(min(max(w, eps), 1.0))
    raise ValueError(mode)


def blend(mg: MemoryGraph, j: int, goal: int, mu: float) -> float:
    """Value in [0,1]: mu * semantic distance to goal + (1-mu) * emotional dissatisfaction."""
    logical = (1.0 - cosine(mg.nodes[j].emb, mg.nodes[goal].emb)) / 2.0          # [0,1]
    emotional = (1.0 - mg.nodes[j].valence) / 2.0                                  # [0,1]
    return mu * logical + (1.0 - mu) * emotional


def astar_path(mg: MemoryGraph, start: int, goal: int, heuristic: str = "blend_scaled",
               mu: float = 0.7, cost_mode: str = "inverse", eps: float = 1e-6,
               h_scale: float = 10.0):
    """Returns (path, total_cost, n_expanded). path is [] if goal unreachable."""
    G = mg.G
    if start == goal:
        return [start], 0.0, 0
    costs = {(i, j): edge_cost(d.get("W", 0.0), eps, cost_mode) for i, j, d in G.edges(data=True)}
    c_min = min(costs.values()) if costs else 0.0

    def h(j: int) -> float:
        if heuristic == "zero" or j == goal:
            return 0.0
        b = blend(mg, j, goal, mu)
        if heuristic == "blend_scaled":
            return c_min * b      # <= c_min <= cost of any remaining (>=1) edge  => admissible, consistent
        if heuristic == "blend_raw":
            return h_scale * b  # deliberately unscaled; can overestimate
        raise ValueError(heuristic)

    openh = [(h(start), 0.0, start)]
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while openh:
        f, gc, u = heapq.heappop(openh)
        if u in closed:
            continue
        if u == goal:
            path = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1], gc, expanded
        closed.add(u)
        expanded += 1
        for v in G.successors(u):
            ng = gc + costs[(u, v)]
            if ng < g.get(v, math.inf):
                g[v] = ng
                parent[v] = u
                heapq.heappush(openh, (ng + h(v), ng, v))
    return [], math.inf, expanded
