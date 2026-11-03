"""R-2026-11-03-02: confirmatory test of RRF(BM25, cosine) vs cosine and vs BM25 on LoCoMo (not used to discover RRF's gain).
Usage: python experiments/locomo_rrf.py --run-id R-2026-11-03-02"""
import argparse, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from locomo_retr import *
from conslayer.runlog import Run
from conslayer import stats as st

PRIMARY = "all_ev@10"
POP = "categories 1-4 (multi-hop, temporal, open-domain, single-hop) with >=1 valid evidence turn"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); a = ap.parse_args()
    cfg = dict(dataset_sha256=L.sha256_file(L.DATA), emb_sha256=L.sha256_file(L.EMB), embedder=L.EMBEDDER, population=POP,
               primary_metric=PRIMARY, primary_comparisons=["rrf - cosine", "rrf - bm25"], rrf_k=60,
               tests="two-sided paired sign-flip (B=20000, seed 0), Holm over the 2 primary comparisons; paired percentile bootstrap CI (B=10000, seed 0) over questions",
               confirm_rule="CONFIRMED iff both diffs > 0 and both Holm-adjusted p < 0.05", bm25="rank_bm25.BM25Okapi defaults, [a-z0-9]+ lowercase tokens",
               sensitivity="conversation-cluster bootstrap (10 clusters) reported, not used for the decision")
    run = Run(a.run_id, cfg, seed=0, models={"embedder": L.EMBEDDER})
    E, idx = L.load_cache(); rows = []
    for conv in L.load_convs():
        ci = ConvIndex(conv, E, idx)
        for q in conv["qa"]:
            if q["category"] == 5 or q["n_ev_valid"] < 1:
                continue
            fs = ci.first_stages(q["question"]); gold = set(q["gold"])
            for name, r in (("cosine", fs["r_cos"]), ("bm25", fs["r_bm"]), ("rrf", fs["r_rrf"])):
                rows.append(dict(qid=q["qid"], conv=conv["sample_id"], category=q["category"], n_ev=q["n_ev_valid"], retriever=name, **metrics(r, gold)))
    df = pd.DataFrame(rows); run.save_csv("per_question.csv", df)
    mets = [c for c in df.columns if "@" in c]
    piv = {m: df.pivot(index="qid", columns="retriever", values=m) for m in mets}
    meta = df[df.retriever == "rrf"].set_index("qid")[["conv", "category", "n_ev"]]
    summ, paired = [], []
    subsets = {"all_cat1-4": meta.index, "multi_ev(>=2)": meta.index[meta.n_ev >= 2]}
    for c in (1, 2, 3, 4):
        subsets[f"cat{c}_{L.CATNAME[c]}"] = meta.index[meta.category == c]
    for sname, ids in subsets.items():
        for m in mets:
            for r in ("cosine", "bm25", "rrf"):
                x = piv[m].loc[ids, r].values; mu, lo, hi = st.bootstrap_ci(x, B=10000, seed=0)
                summ.append(dict(subset=sname, n=len(ids), metric=m, retriever=r, mean=mu, lo=lo, hi=hi))
            for other in ("cosine", "bm25"):
                d = piv[m].loc[ids, "rrf"].values - piv[m].loc[ids, other].values
                dd, lo, hi = st.bootstrap_ci(d, B=10000, seed=0)
                paired.append(dict(subset=sname, n=len(ids), metric=m, comparison=f"rrf-{other}", diff=dd, lo=lo, hi=hi, p_signflip=st.sign_flip_p(d, B=20000, seed=0)))
    summ = pd.DataFrame(summ); paired = pd.DataFrame(paired)
    pr = paired[(paired.subset == "all_cat1-4") & (paired.metric == PRIMARY)].copy()
    pr["p_holm"] = st.holm(pr.p_signflip.tolist())
    # conversation-cluster bootstrap (sensitivity)
    rng = np.random.default_rng(0); convs = meta.conv.unique(); cl = []
    for other in ("cosine", "bm25"):
        d = pd.Series(piv[PRIMARY]["rrf"] - piv[PRIMARY][other]); per = d.groupby(meta.conv).mean()
        bs = [per.loc[rng.choice(convs, len(convs))].mean() for _ in range(10000)]
        cl.append(dict(comparison=f"rrf-{other}", conv_mean_of_means=per.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                       n_convs_rrf_better=int((per > 0).sum()), n_convs=len(per)))
    cl = pd.DataFrame(cl)
    pr["confirmed_part"] = (pr["diff"] > 0) & (pr.p_holm < 0.05)
    verdict = "CONFIRMED" if pr.confirmed_part.all() else "NOT CONFIRMED"
    for n, t in (("summary.csv", summ), ("paired.csv", paired), ("primary.csv", pr), ("cluster_sensitivity.csv", cl)):
        run.save_csv(n, t)
    run.finish(summary=dict(verdict=verdict, n_questions=int(len(meta)), primary=pr.round(4).to_dict("records"), cluster=cl.round(4).to_dict("records")))
    print(summ[(summ.subset == "all_cat1-4") & summ.metric.isin(["all_ev@10", "all_ev@5", "any@10", "frac@10"])].round(3).to_string())
    print(pr.round(4).to_string()); print(cl.round(4).to_string()); print("VERDICT", verdict)


if __name__ == "__main__":
    main()
