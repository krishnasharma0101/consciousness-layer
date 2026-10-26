"""Statistics used by every run: Wilson CI, bootstrap CIs, paired tests, AUROC, Holm correction.
All resampling functions take an explicit seed so the CI in the notebook is reproducible."""
from __future__ import annotations

import math
import numpy as np
from scipy import stats as _st


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(x, B: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, B: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Mean of (a - b) over paired items with a percentile bootstrap CI. Returns (diff, lo, hi)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), B, seed, alpha)


def sign_flip_p(d, B: int = 20000, seed: int = 0) -> float:
    """Two-sided paired permutation (random sign-flip) test of mean(d) == 0."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    s = rng.choice([-1.0, 1.0], size=(B, len(d)))
    null = np.abs((s * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (B + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p from discordant counts b (A right, B wrong) and c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * _st.binom.cdf(min(b, c), n, 0.5)))


def auroc(scores, labels) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)  (Mann-Whitney). labels: 1 = positive."""
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _st.rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auroc_ci(scores, labels, B: int = 5000, seed: int = 0, alpha: float = 0.05):
    """AUROC with a stratified percentile bootstrap CI. Returns (auc, lo, hi)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(bool)
    a = auroc(s, y)
    if math.isnan(a):
        return a, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    out = []
    for _ in range(B):
        ip = rng.choice(pos, len(pos)); ineg = rng.choice(neg, len(neg))
        idx = np.concatenate([ip, ineg])
        out.append(auroc(s[idx], y[idx]))
    return a, float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def holm(pvals) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()
