"""E3b (reduced E4): does the v1 graph help as a RE-RANKING BONUS on top of a strong first stage?
score = (1-w)*minmax(first) + w*minmax(A), A = PPR activation seeded by the first stage's top-5.
First stages: cosine (MiniLM), rrf(bm25, cosine). Grid: W in {v1 (S,T,E), S-only}, zeta in {0.5, 0.85},
w in {0.1, 0.25, 0.5}; each also with shuffled W (control). Dev/test split by sha1(question_id) parity;
config chosen on dev (max turn_any@10, tie nDCG@10, then smaller w), evaluated once on test.
Usage: python experiments/e3b_hybrid.py --run-id R-2026-10-26-04"""
import argparse, hashlib, itertools, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
from conslayer.runlog import Run
from conslayer.theory_v1 import V1Config, V1Graph, rrf
from conslayer.sentiment import valence
from conslayer import stats as st
import lme_common as L
from e3_lme_retrieval import parse_date, bm25_rank, metrics, HALF_LIFE_DAYS

WS = {"v1W": {}, "Sonly": dict(use_T=False, use_E=False)}
ZETAS = (0.5, 0.85); WEIGHTS = (0.1, 0.25, 0.5); FIRSTS = ("cosine", "rrf")


def split(qid):
    return "dev" if int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def one(args):
    x, E_path, idx = args
    E = np.load(E_path, mmap_mode="r")
    texts, times, sess_of, gold = [], [], [], set()
    d0 = parse_date(x["haystack_dates"][0])
    for sid, date, sess in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]):
        base = (parse_date(date) - d0).total_seconds() / 86400
        for ti, t in enumerate(sess):
            if t.get("has_answer"):
                gold.add(len(texts))
            texts.append(t["content"]); times.append(base + ti / 1440.0); sess_of.append(sid)
    V = np.asarray(E[[idx[L.h(t)] for t in texts]], np.float64)
    q = np.asarray(E[idx[L.h(x["question"])]], np.float64)
    vals = np.array([valence(t) for t in texts]); gs = set(x["answer_session_ids"])
    cos = V @ q
    first = {"cosine": cos, "rrf": rrf([np.argsort(-cos, kind="stable"), bm25_rank(texts, x["question"])], n=len(texts))}
    common = dict(qid=x["question_id"], qtype=x["question_type"], split=split(x["question_id"]))
    rows = [dict(common, first=f, W="-", zeta=np.nan, w=0.0, control="none", **metrics(np.argsort(-s, kind="stable"), gold, sess_of, gs))
            for f, s in first.items()]
    for (wn, over), zeta, control in itertools.product(WS.items(), ZETAS, ("none", "shuffle")):
        g = V1Graph(V, times, vals, V1Config(half_life=HALF_LIFE_DAYS, time_unit="days", zeta=zeta, control=control, **over))
        for f, s in first.items():
            for w in WEIGHTS:
                rows.append(dict(common, first=f, W=wn, zeta=zeta, w=w, control=control,
                                 **metrics(g.hybrid_rank(s, w), gold, sess_of, gs)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); ap.add_argument("--procs", type=int, default=7)
    a = ap.parse_args()
    cfg = dict(dataset_sha256=L.sha256_file(L.DATA), emb_sha256=L.sha256_file(L.EMB), first_stages=FIRSTS, rrf_k=60,
               W_variants=WS, zetas=ZETAS, weights=WEIGHTS, seed_k=5, half_life_days=HALF_LIFE_DAYS,
               split="sha1(question_id) % 2 == 0 -> dev", selection="dev: max turn_any@10, tie nDCG@10, then smaller w",
               primary="test turn_any@10, chosen hybrid vs its own first stage, paired; Holm over 2 first stages",
               bootstrap_B=10000)
    run = Run(a.run_id, cfg, seed=0, models={"embedder": L.EMBEDDER})
    data = [x for x in L.load() if "_abs" not in x["question_id"]]
    idx = L.load_cache()[1]; t0 = time.time(); rows = []
    with Pool(a.procs) as p:
        for rr in p.imap(one, [(x, L.EMB, idx) for x in data], chunksize=2):
            rows += rr
    df = pd.DataFrame(rows); run.save_csv("per_question.csv", df)
    key = ["first", "W", "zeta", "w", "control"]
    df["zeta"] = df["zeta"].fillna(-1)
    grid = df.groupby(["split"] + key)[["turn_any@10", "turn_ndcg@10", "sess_all@10"]].mean().reset_index()
    run.save_csv("grid_means.csv", grid)
    out, prim = [], []
    for f in FIRSTS:
        dev = grid[(grid.split == "dev") & (grid["first"] == f) & (grid.control == "none") & (grid.w > 0)]
        best = dev.sort_values(["turn_any@10", "turn_ndcg@10", "w"], ascending=[False, False, True]).iloc[0]
        sel = {k: best[k] for k in key}
        test = df[df.split == "test"]
        base = test[(test["first"] == f) & (test.w == 0)].set_index("qid")
        for ctl in ("none", "shuffle"):
            h = test[(test["first"] == f) & (test.W == sel["W"]) & (test.zeta == sel["zeta"]) & (test.w == sel["w"]) & (test.control == ctl)].set_index("qid")
            for m in ["turn_any@10", "turn_any@5", "turn_ndcg@10", "sess_all@10"]:
                for sub, ids in (("all", h.index), ("multi-session", h.index[h.qtype == "multi-session"])):
                    d = h.loc[ids, m].values - base.loc[ids, m].values
                    d = d[~np.isnan(d)]
                    mm, lo, hi = st.bootstrap_ci(d, B=10000, seed=0)
                    r = dict(first=f, W=sel["W"], zeta=sel["zeta"], w=sel["w"], control=ctl, metric=m, subset=sub, n=len(d),
                             base=base.loc[ids, m].mean(), hybrid=h.loc[ids, m].mean(), diff=mm, lo=lo, hi=hi,
                             p_signflip=st.sign_flip_p(d, B=20000, seed=0), dev_best_turn_any10=best["turn_any@10"])
                    out.append(r)
                    if ctl == "none" and m == "turn_any@10" and sub == "all":
                        prim.append(r)
    out = pd.DataFrame(out); prim = pd.DataFrame(prim); prim["p_holm"] = st.holm(prim.p_signflip.values)
    run.save_csv("test_results.csv", out); run.save_csv("primary_holm.csv", prim)
    run.finish(summary=dict(n=len(data), n_dev=int((df[df.w == 0].split == "dev").sum() // 2), wall_s=round(time.time() - t0, 1)))
    pd.set_option("display.width", 250)
    print(grid[grid.control == "none"].pivot_table(index=key[:4], columns="split", values="turn_any@10").round(3).to_string())
    print(out.round(4).to_string()); print(prim.round(4).to_string())
