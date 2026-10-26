"""E2: is v1 C_norm predictive of failure?  (no LLM calls)
Part A (synthetic v0, R-2026-10-20-01 answers): recompute v1 C_norm per probe; label = wrong for
  sandwich_always / sandwich_nodraft / rag; score = -C_norm (low C -> wrong, paper §5.2).
Part B (LongMemEval-S, E3 per_question.csv): label = retrieval miss (turn_any@10 == 0) for v1 and cosine;
  score = -C_norm (v1 graph); baselines: -max cosine(question, turn), -Phi_dyn.
Usage: python experiments/e2_cnorm_signal.py --run-id R-... --e3 results/R-2026-10-26-02"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
from conslayer.runlog import Run
from conslayer.theory_v1 import V1Config, V1Graph
from conslayer.sentiment import valence
from conslayer.synthetic import make_dialogues
from conslayer import stats as st
import lme_common as L

B = 5000


def auc_row(part, label_name, score_name, s, y, **kw):
    a, lo, hi = st.auroc_ci(s, y, B=B, seed=0)
    return dict(part=part, label=label_name, score=score_name, n=len(y), n_pos=int(np.sum(y)), auroc=a, lo=lo, hi=hi, **kw)


def resid(x, z):
    X = np.c_[np.ones(len(z)), z]
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    return x - X @ beta


def part_a(rows):
    from conslayer.systems import Embedder
    emb = Embedder(cache_dir=L.MODELS)
    pp = pd.read_csv("results/v0/per_probe.csv"); nd = pd.read_csv("results/v0/per_probe_nodraft.csv")
    dia = make_dialogues(40, 0)
    recs = []
    for d in dia:
        E = emb(d["turns"]); vals = [valence(t) for t in d["turns"]]
        g = V1Graph(E, list(range(len(d["turns"]))), vals, V1Config())
        for p in d["probes"]:
            draft = pp[(pp.dialogue == d["id"]) & (pp.ptype == p["type"]) & (pp.system == "plain")].answer.iloc[0]
            draft = "" if pd.isna(draft) else str(draft)
            r = {"dialogue": d["id"], "ptype": p["type"]}
            for cue_name, cue in (("q+draft", p["q"] + " " + draft), ("q", p["q"])):
                c = emb([cue])[0]
                res = g.retrieve(c, k=4, t_now=len(d["turns"]))
                r[f"c_norm[{cue_name}]"] = res["c_norm"]; r[f"phi_dyn[{cue_name}]"] = res["phi_dyn"]
                r[f"maxcos[{cue_name}]"] = float(np.max(E @ c))
            for sysname, df in (("sandwich_always", pp), ("rag", pp), ("sandwich_nodraft", nd)):
                r[f"wrong[{sysname}]"] = int(1 - df[(df.dialogue == d["id"]) & (df.ptype == p["type"]) & (df.system == sysname)].correct.iloc[0])
            r["v0_c_norm"] = pp[(pp.dialogue == d["id"]) & (pp.ptype == p["type"]) & (pp.system == "sandwich_always")].c_norm.iloc[0]
            recs.append(r)
    A = pd.DataFrame(recs)
    for lab in ["sandwich_always", "rag", "sandwich_nodraft"]:
        y = A[f"wrong[{lab}]"].values
        for sc in ["c_norm[q+draft]", "c_norm[q]", "phi_dyn[q+draft]", "maxcos[q+draft]", "maxcos[q]", "v0_c_norm"]:
            rows.append(auc_row("A_synthetic", f"wrong[{lab}]", "-" + sc, -A[sc].values, y))
    return A


def part_b(rows, e3dir):
    df = pd.read_csv(os.path.join(e3dir, "per_question.csv"))
    data = {x["question_id"]: x for x in L.load()}
    E, idx = L.load_cache()
    mc = {}
    for qid in df.qid.unique():
        x = data[qid]
        V = np.asarray(E[[idx[L.h(t["content"])] for s in x["haystack_sessions"] for t in s]], np.float64)
        mc[qid] = float(np.max(V @ np.asarray(E[idx[L.h(x["question"])]], np.float64)))
    v1 = df[df.retriever == "v1"].set_index("qid")
    v1["maxcos"] = pd.Series(mc)
    out = v1[["qtype", "c_norm", "phi_dyn", "maxcos", "start_ne_goal", "goal_is_gold"]].copy()
    for lab in ["v1", "cosine", "bm25"]:
        miss = (df[df.retriever == lab].set_index("qid").loc[out.index, "turn_any@10"] == 0).astype(int)
        out[f"miss[{lab}]"] = miss
        y = miss.values
        for sc in ["c_norm", "phi_dyn", "maxcos"]:
            rows.append(auc_row("B_longmemeval", f"miss@10[{lab}]", "-" + sc, -out[sc].values, y))
        rows.append(auc_row("B_longmemeval", f"miss@10[{lab}]", "-resid(c_norm|maxcos)", -resid(out.c_norm.values, out.maxcos.values), y))
        for qt, g in out.groupby("qtype"):
            if g[f"miss[{lab}]"].nunique() == 2:
                rows.append(auc_row("B_by_qtype", f"miss@10[{lab}]", "-c_norm", -g.c_norm.values, g[f"miss[{lab}]"].values, qtype=qt))
    return out.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); ap.add_argument("--e3", required=True)
    a = ap.parse_args()
    run = Run(a.run_id, dict(e3_dir=a.e3, auroc_bootstrap_B=B, positive_class="failure (wrong answer / retrieval miss)",
                             score_direction="-C_norm: low C_norm predicts failure (paper §5.2)",
                             partA_graph=V1Config().to_dict(), partA_k=4, partB_graph="v1 config of the E3 run"),
              seed=0, models={"embedder": L.EMBEDDER})
    rows = []
    A = part_a(rows); run.save_csv("partA_per_probe.csv", A)
    Bdf = part_b(rows, a.e3); run.save_csv("partB_per_question.csv", Bdf)
    R = pd.DataFrame(rows); run.save_csv("auroc.csv", R)
    run.finish(summary=dict(n_partA=len(A), n_partB=len(Bdf)))
    pd.set_option("display.width", 200)
    print(R.round(3).to_string())
    print(A.groupby("ptype")[["c_norm[q+draft]", "c_norm[q]", "wrong[sandwich_always]", "wrong[rag]"]].mean().round(4))
