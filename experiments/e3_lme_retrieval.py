"""E3: LongMemEval-S retrieval recall@k, no LLM.
Retrievers: BM25, cosine (MiniLM), theory-v1 graph (activation + A*), controls (shuffled W, random topology),
ablations (-S, -T, -E, -R, -A*), exploratory (+recency, zeta sweep).
Usage: python experiments/e3_lme_retrieval.py --run-id R-2026-10-26-02 [--limit N]"""
import argparse, os, re, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from datetime import datetime
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from conslayer.runlog import Run
from conslayer.theory_v1 import V1Config, V1Graph
from conslayer.sentiment import valence
from conslayer import stats as st
import lme_common as L

KS = (1, 5, 10)
HALF_LIFE_DAYS = 7.0
BASE = dict(half_life=HALF_LIFE_DAYS, time_unit="days")
GRAPHS = {
    "v1": {},
    "v1_shuffledW": dict(control="shuffle"),
    "v1_randomtopo": dict(control="random_topology"),
    "v1_noS": dict(use_S=False),
    "v1_noT": dict(use_T=False),
    "v1_noE": dict(use_E=False),
    "v1_noR": dict(use_R=False),
    "v1_noAstar": dict(use_astar=False),
    # exploratory (not part of the confirmatory family)
    "x_v1_recency": dict(use_recency=True, recency_half_life=30.0),
    "x_v1_zeta0.5": dict(zeta=0.5),
    "x_v1_zeta0.15": dict(zeta=0.15),
}
CONFIRM = ["v1", "v1_shuffledW", "v1_randomtopo", "v1_noS", "v1_noT", "v1_noE", "v1_noR", "v1_noAstar"]


def parse_date(s):
    return datetime.strptime(s.split(" (")[0] + " " + s.split(") ")[1], "%Y/%m/%d %H:%M")


TOK = re.compile(r"[a-z0-9]+")


def bm25_rank(docs, query):
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([TOK.findall(d.lower()) or ["_"] for d in docs])
    return np.argsort(-bm.get_scores(TOK.findall(query.lower())), kind="stable")


def metrics(ranked, gold_turns, sess_of, gold_sess):
    out = {}
    for k in KS:
        top = list(ranked[:k])
        if gold_turns:
            hit = [i in gold_turns for i in top]
            out[f"turn_any@{k}"] = float(any(hit))
            out[f"turn_frac@{k}"] = sum(hit) / len(gold_turns)
        ss = {sess_of[i] for i in top}
        out[f"sess_any@{k}"] = float(bool(ss & gold_sess))
        out[f"sess_all@{k}"] = float(gold_sess <= ss)
    if gold_turns:
        top = list(ranked[:10])
        dcg = sum(1 / np.log2(r + 2) for r, i in enumerate(top) if i in gold_turns)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(10, len(gold_turns))))
        out["turn_ndcg@10"] = dcg / idcg
    return out


def one_question(args):
    x, E_path, idx = args
    E = np.load(E_path, mmap_mode="r")
    texts, times, sess_of, gold_turns = [], [], [], set()
    d0 = parse_date(x["haystack_dates"][0])
    for si, (sid, date, sess) in enumerate(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"])):
        base = (parse_date(date) - d0).total_seconds() / 86400
        for ti, t in enumerate(sess):
            if t.get("has_answer"):
                gold_turns.add(len(texts))
            texts.append(t["content"]); times.append(base + ti / 1440.0); sess_of.append(sid)
    V = np.asarray(E[[idx[L.h(t)] for t in texts]], np.float64)
    q = np.asarray(E[idx[L.h(x["question"])]], np.float64)
    t_now = (parse_date(x["question_date"]) - d0).total_seconds() / 86400
    vals = np.array([valence(t) for t in texts])
    gold_sess = set(x["answer_session_ids"])
    rows, diag = [], {}
    common = dict(qid=x["question_id"], qtype=x["question_type"], n_nodes=len(texts), n_gold_turns=len(gold_turns))
    t0 = time.perf_counter(); r = bm25_rank(texts, x["question"])
    rows.append(dict(common, retriever="bm25", sec=time.perf_counter() - t0, **metrics(r, gold_turns, sess_of, gold_sess)))
    t0 = time.perf_counter(); r = np.argsort(-(V @ q), kind="stable")
    rows.append(dict(common, retriever="cosine", sec=time.perf_counter() - t0, **metrics(r, gold_turns, sess_of, gold_sess)))
    for name, over in GRAPHS.items():
        t0 = time.perf_counter()
        g = V1Graph(V, times, vals, V1Config(**{**BASE, **over}))
        res = g.retrieve(q, k=max(KS), t_now=t_now)
        extra = dict(c_norm=res["c_norm"], phi_dyn=res["phi_dyn"], n_iter=res["n_iter"], path_len=len(res["path"]),
                     start_ne_goal=int(res["start"] != res["goal"]), n_edges=int(g.mask.sum() // 2),
                     goal_is_gold=int(res["goal"] in gold_turns))
        rows.append(dict(common, retriever=name, sec=time.perf_counter() - t0, **extra,
                         **metrics(res["ids"], gold_turns, sess_of, gold_sess)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--procs", type=int, default=7)
    a = ap.parse_args()
    cfg = dict(dataset="longmemeval_s_cleaned.json", dataset_sha256=L.sha256_file(L.DATA), emb_sha256=L.sha256_file(L.EMB),
               embedder=L.EMBEDDER, ks=KS, nodes="all turns (user+assistant), haystack order",
               time="days since first haystack session + turn_index/1440", half_life_days=HALF_LIFE_DAYS,
               cue="question embedding only (no LLM draft)", exclude="abstention questions (_abs)",
               valence="conslayer.sentiment lexicon", bm25="rank_bm25.BM25Okapi defaults, lowercase [a-z0-9]+ tokens",
               graph_base=V1Config(**BASE).to_dict(), graph_variants=GRAPHS, confirmatory=CONFIRM,
               primary_metric="turn_any@10 (questions with >=1 has_answer turn)", bootstrap_B=10000, bootstrap_seed=0)
    run = Run(a.run_id, cfg, seed=0, models={"embedder": L.EMBEDDER})
    data = [x for x in L.load() if "_abs" not in x["question_id"]]
    if a.limit:
        data = data[: a.limit]; run.note(f"LIMIT {a.limit} questions (smoke test)")
    idx = L.load_cache()[1]
    t0 = time.time(); rows = []
    with Pool(a.procs) as p:
        for i, rr in enumerate(p.imap(one_question, [(x, L.EMB, idx) for x in data], chunksize=2)):
            rows += rr
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(data)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    run.save_csv("per_question.csv", df)
    # ---- summary with bootstrap CIs and paired tests vs cosine ----
    mets = [c for c in df.columns if "@" in c]
    summ, paired = [], []
    cos = df[df.retriever == "cosine"].set_index("qid")
    for name, g in df.groupby("retriever", sort=False):
        g = g.set_index("qid")
        for m in mets:
            v = g[m].dropna()
            mean, lo, hi = st.bootstrap_ci(v.values, B=10000, seed=0)
            summ.append(dict(retriever=name, metric=m, n=len(v), mean=mean, lo=lo, hi=hi))
            if name != "cosine":
                c = cos.loc[v.index, m]
                dlt = v.values - c.values
                dd, dlo, dhi = st.bootstrap_ci(dlt, B=10000, seed=0)
                paired.append(dict(retriever=name, metric=m, n=len(v), diff_vs_cosine=dd, lo=dlo, hi=dhi,
                                   p_signflip=st.sign_flip_p(dlt, B=20000, seed=0), wins=int((dlt > 0).sum()), losses=int((dlt < 0).sum())))
    summ = pd.DataFrame(summ); paired = pd.DataFrame(paired)
    prim = paired[(paired.metric == "turn_any@10") & paired.retriever.isin(CONFIRM)].copy()
    prim["p_holm"] = st.holm(prim.p_signflip.values)
    run.save_csv("summary.csv", summ); run.save_csv("paired_vs_cosine.csv", paired); run.save_csv("primary_holm.csv", prim)
    bytype = df.groupby(["retriever", "qtype"])[["turn_any@10", "sess_any@10", "turn_frac@10"]].mean().reset_index()
    run.save_csv("by_qtype.csv", bytype)
    # ablation vs full v1 (paired)
    v1 = df[df.retriever == "v1"].set_index("qid")
    abl = []
    for name in CONFIRM[1:] + [k for k in GRAPHS if k.startswith("x_")]:
        g = df[df.retriever == name].set_index("qid")
        for m in ["turn_any@10", "turn_any@5", "sess_any@10", "turn_ndcg@10"]:
            v = g[m].dropna(); dlt = v.values - v1.loc[v.index, m].values
            dd, lo, hi = st.bootstrap_ci(dlt, B=10000, seed=0)
            abl.append(dict(variant=name, metric=m, diff_vs_v1=dd, lo=lo, hi=hi, p_signflip=st.sign_flip_p(dlt, B=20000, seed=0)))
    run.save_csv("ablation_vs_v1.csv", abl)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    s = summ[summ.metric == "turn_any@10"].reset_index(drop=True)
    ax.bar(s.retriever, s["mean"], yerr=[s["mean"] - s.lo, s.hi - s["mean"]], capsize=3)
    ax.set_ylabel("turn recall_any@10"); ax.set_ylim(0, 1); plt.xticks(rotation=45, ha="right"); ax.set_title(a.run_id)
    run.savefig("recall_at10.png", fig)
    sec = df.groupby("retriever").sec.mean().to_dict()
    run.finish(summary=dict(n_questions=int(df.qid.nunique()), n_questions_turn_level=int(df[df.retriever == "cosine"]["turn_any@10"].notna().sum()),
                            mean_sec_per_question=sec, wall_s=round(time.time() - t0, 1)))
    print(summ[summ.metric.isin(["turn_any@10", "turn_any@5", "turn_any@1", "sess_any@10"])].pivot(index="retriever", columns="metric", values="mean").round(3).to_string())
    print(prim.round(4).to_string())
