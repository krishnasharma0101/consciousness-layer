"""R-2026-11-03-03: LoCoMo multi-hop evidence retrieval; pre-registered test 'v1 graph beats RRF by >= 0.03 on all-evidence@10'.
Usage: python experiments/locomo_graph.py --run-id R-2026-11-03-03 [--limit-convs N]"""
import argparse, itertools, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from locomo_retr import *
from conslayer.runlog import Run
from conslayer import stats as st

PRIMARY = "all_ev@10"; MARGIN = 0.03
FIRST = {  # first-stage graph retrievers (ranking = v1 retrieve(): goal, A* path, then activation order)
    "v1": {}, "v1_shuffledW": dict(control="shuffle"), "v1_randomtopo": dict(control="random_topology"),
    "v1_noS": dict(use_S=False), "v1_noT": dict(use_T=False), "v1_noE": dict(use_E=False), "v1_noR": dict(use_R=False),
    "v1_noAstar": dict(use_astar=False)}
STRONG = dict(zeta=0.85)
RERANK = {  # name -> (graph overrides, w).  Graph-as-rerank over the RRF score.
    "rr_sonly": (dict(SONLY, zeta=0.5), 0.1),                       # PRIMARY (config transferred from the LongMemEval dev selection)
    "rr_sonly_shuf": (dict(SONLY, zeta=0.5, control="shuffle", control_seed=0), 0.1),
    "rr_sonly_shuf_s1": (dict(SONLY, zeta=0.5, control="shuffle", control_seed=1), 0.1),
    "rr_sonly_shuf_s2": (dict(SONLY, zeta=0.5, control="shuffle", control_seed=2), 0.1),
    "rr_sonly_rand": (dict(SONLY, zeta=0.5, control="random_topology"), 0.1),
    "rr_v1W": (dict(zeta=0.5), 0.1),
    "rr_strong": (dict(STRONG), 0.5),                               # aggressive config: full W, zeta 0.85, w 0.5
    "rr_strong_shuf": (dict(STRONG, control="shuffle"), 0.5),
    "rr_strong_rand": (dict(STRONG, control="random_topology"), 0.5),
    "rr_strong_noS": (dict(STRONG, use_S=False), 0.5), "rr_strong_noT": (dict(STRONG, use_T=False), 0.5),
    "rr_strong_noE": (dict(STRONG, use_E=False), 0.5), "rr_strong_noR": (dict(STRONG, use_R=False), 0.5)}
FAMILY = [("rrf", "cosine"), ("rrf", "bm25"), ("v1", "rrf"), ("rr_strong", "rrf"), ("rr_v1W", "rrf"),
          ("rr_sonly", "rr_sonly_shuf"), ("rr_sonly", "rr_sonly_rand"), ("rr_strong", "rr_strong_shuf"), ("rr_strong", "rr_strong_rand"),
          ("v1", "v1_shuffledW"), ("v1", "v1_randomtopo")] + [(f"v1_no{x}", "v1") for x in ("S", "T", "E", "R", "Astar")] \
         + [(f"rr_strong_no{x}", "rr_strong") for x in ("S", "T", "E", "R")]
PRIMARY_PAIR = ("rr_sonly", "rrf")


def one_conv(args):
    ci_i, = args
    conv = L.load_convs()[ci_i]; E, idx = L.load_cache(); ci = ConvIndex(conv, E, idx); rows = []
    for q in conv["qa"]:
        if q["category"] == 5 or q["n_ev_valid"] < 1:
            continue
        fs = ci.first_stages(q["question"]); gold = set(q["gold"]); qe = ci.q_emb(q["question"])
        base = dict(qid=q["qid"], conv=conv["sample_id"], category=q["category"], n_ev=q["n_ev_valid"])
        for name, r in (("cosine", fs["r_cos"]), ("bm25", fs["r_bm"]), ("rrf", fs["r_rrf"])):
            rows.append(dict(base, retriever=name, **metrics(r, gold)))
        for name, over in FIRST.items():
            g = ci.graph(name, over); res = g.retrieve(qe, k=10, t_now=float(ci.times.max()))
            rows.append(dict(base, retriever=name, c_norm=res["c_norm"], goal_is_gold=int(res["goal"] in gold), **metrics(res["ids"], gold)))
        for name, (over, w) in RERANK.items():
            g = ci.graph(name, over)
            rows.append(dict(base, retriever=name, **metrics(g.hybrid_rank(fs["s_rrf"], w), gold)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit-convs", type=int, default=0); a = ap.parse_args()
    cfg = dict(dataset_sha256=L.sha256_file(L.DATA), emb_sha256=L.sha256_file(L.EMB), embedder=L.EMBEDDER, ks=KS,
               half_life_days=HALF_LIFE_DAYS, graph_base=V1Config(**BASE).to_dict(), first_stage_graphs=FIRST,
               rerank={k: dict(overrides=v[0], w=v[1]) for k, v in RERANK.items()}, rrf_k=60, t_now="max node time (only used if use_recency)",
               primary_metric=PRIMARY, primary_subset="category 1 (multi-hop) with >=2 valid evidence turns", margin=MARGIN,
               primary_pair=PRIMARY_PAIR, family_holm=FAMILY, bootstrap="paired percentile, B=10000, seed 0 over questions",
               sign_flip="two-sided paired, B=20000, seed 0")
    run = Run(a.run_id, cfg, seed=0, models={"embedder": L.EMBEDDER})
    n = len(L.load_convs()); order = list(range(n))[: a.limit_convs or n]
    order = sorted(order, key=lambda i: -i)  # no particular reason; keep deterministic
    t0 = time.time(); rows = []
    with Pool(min(a.procs, len(order))) as p:
        for rr in p.imap_unordered(one_conv, [(i,) for i in order]):
            rows += rr; print(f"{len(rows)} rows {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); run.save_csv("per_question.csv", df)
    mets = [c for c in df.columns if "@" in c]
    piv = {m: df.pivot(index="qid", columns="retriever", values=m) for m in mets}
    meta = df[df.retriever == "rrf"].set_index("qid")[["conv", "category", "n_ev"]]
    subsets = {"cat1_multihop_ev>=2 [PRIMARY]": meta.index[(meta.category == 1) & (meta.n_ev >= 2)],
               "all_cat1-4_ev>=2": meta.index[meta.n_ev >= 2]}
    for c in (1, 2, 3, 4):
        subsets[f"cat{c}_{L.CATNAME[c]}_ev>=2"] = meta.index[(meta.category == c) & (meta.n_ev >= 2)]
        subsets[f"cat{c}_{L.CATNAME[c]}_all"] = meta.index[meta.category == c]
    summ, paired = [], []
    for sname, ids in subsets.items():
        for m in mets:
            for r in sorted(df.retriever.unique()):
                mu, lo, hi = st.bootstrap_ci(piv[m].loc[ids, r].values, B=10000, seed=0)
                summ.append(dict(subset=sname, n=len(ids), metric=m, retriever=r, mean=mu, lo=lo, hi=hi))
        for m in (PRIMARY, "all_ev@5", "frac@10", "any@10"):
            for x, y in [PRIMARY_PAIR] + FAMILY:
                d = piv[m].loc[ids, x].values - piv[m].loc[ids, y].values
                dd, lo, hi = st.bootstrap_ci(d, B=10000, seed=0)
                paired.append(dict(subset=sname, n=len(ids), metric=m, a=x, b=y, diff=dd, lo=lo, hi=hi, p_signflip=st.sign_flip_p(d, B=20000, seed=0)))
    summ = pd.DataFrame(summ); paired = pd.DataFrame(paired)
    P = "cat1_multihop_ev>=2 [PRIMARY]"
    fam = paired[(paired.subset == P) & (paired.metric == PRIMARY) & ~((paired.a == PRIMARY_PAIR[0]) & (paired.b == PRIMARY_PAIR[1]))].copy()
    fam["p_holm"] = st.holm(fam.p_signflip.tolist())
    prim = paired[(paired.subset == P) & (paired.metric == PRIMARY) & (paired.a == PRIMARY_PAIR[0]) & (paired.b == PRIMARY_PAIR[1])].iloc[0]
    shuf = paired[(paired.subset == P) & (paired.metric == PRIMARY) & (paired.a == "rr_sonly") & (paired.b == "rr_sonly_shuf")].iloc[0]
    fal_a = bool(prim.hi < MARGIN); fal_b = bool(shuf["diff"] <= 0)
    verdict = "FALSIFIED" if (fal_a or fal_b) else ("SUPPORTED" if (prim["diff"] >= MARGIN and prim.lo > 0) else "INCONCLUSIVE")
    for nme, t in (("summary.csv", summ), ("paired_all.csv", paired), ("family_holm.csv", fam)):
        run.save_csv(nme, t)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    s = summ[(summ.subset == P) & (summ.metric == PRIMARY)].sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 6)); ax.barh(s.retriever, s["mean"], xerr=[s["mean"] - s.lo, s.hi - s["mean"]], capsize=2)
    ax.set_xlabel("all-evidence@10 (cat 1, >=2 evidence turns)"); run.savefig("all_ev_at10_cat1.png", fig)
    run.finish(summary=dict(verdict=verdict, falsifier_A_ci_upper_below_margin=fal_a, falsifier_B_graph_le_shuffled=fal_b,
                            primary=dict(diff=prim["diff"], lo=prim.lo, hi=prim.hi, p=prim.p_signflip), n_primary=int(len(subsets[P])),
                            wall_s=round(time.time() - t0, 1)))
    print(summ[(summ.subset == P) & (summ.metric.isin([PRIMARY, "all_ev@5"]))].pivot(index="retriever", columns="metric", values="mean").round(3).to_string())
    print(prim.to_string()); print(fam.round(4).to_string()); print("VERDICT", verdict, fal_a, fal_b)


if __name__ == "__main__":
    main()
