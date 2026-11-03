"""LoCoMo retrieval helpers shared by R-2026-11-03-02/-03/-04. Same BM25/RRF/graph code as the LongMemEval runs."""
import os, sys, re
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
from conslayer.theory_v1 import V1Config, V1Graph, rrf
from conslayer.sentiment import valence
import locomo_common as L

KS = (5, 10)
HALF_LIFE_DAYS = 7.0
BASE = dict(half_life=HALF_LIFE_DAYS, time_unit="days")
TOK = re.compile(r"[a-z0-9]+")
SONLY = dict(use_T=False, use_E=False)


class ConvIndex:
    """Per-conversation BM25 + embedding matrix (built once per worker)."""
    def __init__(self, conv, E, idx):
        from rank_bm25 import BM25Okapi
        self.conv = conv
        self.V = np.asarray(E[[idx[L.h(t)] for t in conv["texts"]]], np.float64)
        self.times = np.asarray(conv["times"], float)
        self.vals = np.array([valence(t) for t in conv["texts"]])
        self.bm = BM25Okapi([TOK.findall(t.lower()) or ["_"] for t in conv["texts"]])
        self.E, self.idx = E, idx
        self._g = {}

    def q_emb(self, q):
        return np.asarray(self.E[self.idx[L.h(q)]], np.float64)

    def graph(self, name, over):
        if name not in self._g:
            self._g[name] = V1Graph(self.V, self.times, self.vals, V1Config(**{**BASE, **over}))
        return self._g[name]

    def first_stages(self, q):
        cos = self.V @ self.q_emb(q)
        bm = self.bm.get_scores(TOK.findall(q.lower()))
        r_cos = np.argsort(-cos, kind="stable"); r_bm = np.argsort(-bm, kind="stable")
        s_rrf = rrf([r_cos, r_bm], n=len(cos))
        return dict(cos=cos, bm=bm, r_cos=r_cos, r_bm=r_bm, s_rrf=s_rrf, r_rrf=np.argsort(-s_rrf, kind="stable"))


def metrics(ranked, gold):
    """gold: set of node indices (>=1). all_ev = every gold turn in top-k (primary); frac = recall; any = >=1 gold turn."""
    out = {}
    g = set(gold)
    for k in KS:
        top = set(int(i) for i in list(ranked)[:k])
        inter = len(top & g)
        out[f"all_ev@{k}"] = float(g <= top)
        out[f"frac@{k}"] = inter / len(g)
        out[f"any@{k}"] = float(inter > 0)
    return out
