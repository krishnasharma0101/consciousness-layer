"""Builds the real v1 memory graph for LoCoMo conv-26 with src/conslayer/theory_v1.py and exports it
(plus Python reference results for JS verification) to graph_data.json. Read-only on the repo."""
import json, os, sys, re
import numpy as np, pandas as pd, networkx as nx
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import locomo_common as L
from conslayer.theory_v1 import V1Config, V1Graph
from conslayer.sentiment import valence

CONV = 0
conv = L.load_convs()[CONV]
E, idx = L.load_cache()
texts = conv["texts"]
V = np.asarray(E[[idx[L.h(t)] for t in texts]], np.float64)
t = np.asarray(conv["times"], float)
e = np.array([valence(x) for x in texts])
cfg = V1Config(half_life=7.0, time_unit="days")   # same BASE as experiments/locomo_retr.py
g = V1Graph(V, t, e, cfg)
N = g.N
iu = np.where(np.triu(g.mask, 1))
S, T, F, R = g.factors()
edges = [[int(i), int(j), round(float(S[i, j]), 5), round(float(T[i, j]), 5), round(float(F[i, j]), 5)]
         for i, j in zip(*iu)]
perm = np.random.default_rng(0).permutation(len(edges)).tolist()   # shuffled-W control permutation (seed 0)

# layouts: (a) time axis: x = node time (days, scaled), y/z from 2D spring on W; (b) free 3D spring on W
G = nx.Graph()
G.add_nodes_from(range(N))
for i, j, *_ in edges:
    G.add_edge(i, j, weight=float(g.W[i, j]))
sess = [int(re.match(r"D(\d+):", d).group(1)) for d in conv["dia_ids"]]
# (a) time axis: x = turn order grouped by session (gap between sessions); y/z = first two principal components of the
#     MiniLM embeddings (semantically similar turns sit close together vertically)
xs, prev, pos = [], None, 0.0
for i in range(N):
    if prev is not None:
        pos += 1.0 if sess[i] == prev else 8.0
    xs.append(pos); prev = sess[i]
xs = np.array(xs); xs = (xs - xs.min()) / (xs.max() - xs.min()) * 640 - 320
Uc = g.U - g.U.mean(0); _, _, Vt = np.linalg.svd(Uc, full_matrices=False); pc = Uc @ Vt[:3].T
def rs(v, scale):
    lo, hi = np.percentile(v, 2), np.percentile(v, 98); return np.clip((v - (lo + hi) / 2) / ((hi - lo) / 2), -1.3, 1.3) * scale
lay_time = [[round(float(xs[i]), 2), round(float(rs(pc[:, 0], 110)[i]), 2), round(float(rs(pc[:, 1], 110)[i]), 2)] for i in range(N)]
# (b) free layout: 3D spring layout on the unweighted edge set (seed 0)
p3 = nx.spring_layout(G, dim=3, seed=0, weight=None, iterations=400)
arr = np.array([p3[i] for i in range(N)])
lay_free = [[round(float(rs(arr[:, k], 200)[i]), 2) for k in range(3)] for i in range(N)]

nodes = [dict(id=i, dia=conv["dia_ids"][i], spk=conv["speakers"][i], text=texts[i], date=conv["dates"][i],
              t=round(float(t[i]), 4), val=round(float(e[i]), 3), sess=sess[i]) for i in range(N)]

# presets: real LoCoMo questions from this conversation
pq = pd.read_csv(os.path.join(ROOT, "results", "R-2026-11-03-03", "per_question.csv"))
pick = ["conv-26#40", "conv-26#18", "conv-26#52", "conv-26#60", "conv-26#47", "conv-26#23", "conv-26#103", "conv-26#107"]
qa = {q["qid"]: q for q in conv["qa"]}
presets = []
for qid in pick:
    q = qa[qid]
    qe = np.asarray(E[idx[L.h(q["question"])]], np.float64)
    cos = (g.U @ (qe / np.linalg.norm(qe)))
    res = g.retrieve(qe, k=10, t_now=float(t.max()))
    I = g.input_vector(qe); A, traj = g.activate(I, return_traj=True)
    logged = {}
    for r in ("cosine", "bm25", "rrf", "v1", "rr_sonly", "rr_strong"):
        row = pq[(pq.qid == qid) & (pq.retriever == r)]
        if len(row):
            logged[r] = dict(all_ev10=float(row["all_ev@10"].iloc[0]), frac10=round(float(row["frac@10"].iloc[0]), 4))
    presets.append(dict(qid=qid, question=q["question"], answer=str(q["answer"]), category=L.CATNAME[q["category"]],
                        gold=q["gold"], cos=[round(float(c), 5) for c in cos], logged_R_2026_11_03_03=logged,
                        py_ref=dict(top10=[int(x) for x in res["ids"]], c_norm=res["c_norm"], goal=res["goal"], start=res["start"],
                                    path=[int(x) for x in res["path"]], n_iter=res["n_iter"],
                                    c_traj_first5=[g.c_norm(a) for a in traj[:5]])))

# reference values for the ablation/shuffle toggles (first preset), to verify the JS port
q0 = presets[0]
qe0 = np.asarray(E[idx[L.h(q0["question"])]], np.float64)
variants = {}
for name, over in {"full": {}, "noS": dict(use_S=False), "noT": dict(use_T=False), "noE": dict(use_E=False),
                   "noR": dict(use_R=False), "shuffle": dict(control="shuffle")}.items():
    gg = V1Graph(V, t, e, V1Config(**{**cfg.to_dict(), **over}))
    r = gg.retrieve(qe0, k=10)
    variants[name] = dict(c_norm=r["c_norm"], top10=[int(x) for x in r["ids"]], goal=r["goal"])

out = dict(source=dict(dataset="LoCoMo (snap-research/locomo, CC BY-NC 4.0), conversation conv-26 (Caroline & Melanie), all 19 sessions",
                       code="src/conslayer/theory_v1.py V1Graph, V1Config(half_life=7 days) = BASE of experiments/locomo_retr.py",
                       embedder=L.EMBEDDER + " (cached embeddings data/cache/locomo_minilm.npy)",
                       valence="src/conslayer/sentiment.py lexicon", run_for_logged="R-2026-11-03-03 per_question.csv"),
           cfg=cfg.to_dict(), N=N, U=[[round(float(x), 4) for x in row] for row in g.U], n_edges=len(edges), nodes=nodes, edges=edges, shuffle_perm=perm,
           layout_time=lay_time, layout_free=lay_free, presets=presets, py_variants_preset0=variants,
           t_range_days=[float(t.min()), float(t.max())])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "graph_data.json"), "w"), separators=(",", ":"))
print(N, len(edges), {k: (round(v["c_norm"], 5), v["goal"]) for k, v in variants.items()})
for p in presets: print(p["qid"], p["py_ref"]["c_norm"], p["py_ref"]["goal"], p["py_ref"]["path"], p["py_ref"]["n_iter"], p["logged_R_2026_11_03_03"])
