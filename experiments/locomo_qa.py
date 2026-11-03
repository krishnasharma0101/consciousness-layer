"""R-2026-11-03-04: end-to-end QA on LoCoMo multi-hop (cat 1, >=2 evidence turns) with Qwen2.5-1.5B-Instruct, partial-F1.
Arms: Z closed-book; A RRF top-10; B rr_sonly top-10; C Sandwich (v1 first-stage top-10), no draft; D Sandwich with closed-book draft.
Usage: python experiments/locomo_qa.py --run-id R-2026-11-03-04 [--n N]"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from locomo_retr import *
from conslayer.runlog import Run
from conslayer.llm import LocalLLM
from conslayer.locomo_eval import score
from conslayer import stats as st

K = 10
SYS_CTX = ("You answer questions about a conversation between two people, using only the conversation excerpts provided. "
           "Reply with a short answer (a few words). If the answer has several items, list them separated by commas. Do not explain.")
SYS_Z = ("Answer the question about two people's conversation with a short answer (a few words); guess if you are unsure. "
         "If the answer has several items, list them separated by commas. Do not explain.")
RR_SONLY = (dict(SONLY, zeta=0.5), 0.1)


def ctx_text(conv, ids):
    return "\n".join(f"[{conv['dates'][i]}] {conv['texts'][i]}" for i in sorted(ids))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); a = ap.parse_args()
    cfg = dict(dataset_sha256=L.sha256_file(L.DATA), emb_sha256=L.sha256_file(L.EMB), reader=a.model, k=K, max_new_tokens=32, decoding="greedy, bf16, CPU",
               population="category 1, >=2 valid evidence turns, ordered by sha1(qid); n=%s" % (a.n or "all"), system_ctx=SYS_CTX, system_closed=SYS_Z,
               rerank_B=dict(overrides=RR_SONLY[0], w=RR_SONLY[1]), arms=["Z_closed_book", "A_rrf", "B_graph", "C_sandwich_nodraft", "D_sandwich_draft"],
               metric="LoCoMo official partial-F1 (multi-answer for cat 1), conslayer.locomo_eval", primary="B_graph - A_rrf",
               family_holm=["B-A", "C-A", "D-C"], bootstrap="paired percentile B=10000 seed 0")
    run = Run(a.run_id, cfg, seed=0, models={"embedder": L.EMBEDDER, "reader": a.model})
    E, idx = L.load_cache(); items = []
    for conv in L.load_convs():
        ci = None
        for q in conv["qa"]:
            if q["category"] == 1 and q["n_ev_valid"] >= 2:
                items.append((hashlib.sha1(q["qid"].encode()).hexdigest(), conv["sample_id"], q))
    items.sort(key=lambda x: x[0])
    if a.n:
        items = items[: a.n]; run.note(f"n capped to {a.n}")
    convs = {c["sample_id"]: c for c in L.load_convs()}; cis = {}
    llm = LocalLLM(a.model, cache_dir=L.MODELS, threads=os.cpu_count()); cache = {}; ncalls = 0; t0 = time.time(); rows = []; gens = []

    def ask(system, user):
        nonlocal ncalls
        k = hashlib.sha1((system + "\x00" + user).encode()).hexdigest()
        if k not in cache:
            cache[k] = llm.chat(system, user, max_new_tokens=32)[0].strip(); ncalls += 1
        return cache[k]

    for n, (_, sid, q) in enumerate(items):
        conv = convs[sid]
        if sid not in cis:
            cis[sid] = ConvIndex(conv, E, idx)
        ci = cis[sid]; fs = ci.first_stages(q["question"]); qe = ci.q_emb(q["question"])
        A_ids = list(fs["r_rrf"][:K])
        B_ids = list(ci.graph("rr_sonly", RR_SONLY[0]).hybrid_rank(fs["s_rrf"], RR_SONLY[1])[:K])
        C_ids = list(ci.graph("v1", {}).retrieve(qe, k=K, t_now=float(ci.times.max()))["ids"])
        gold = set(q["gold"]); Q = f"Question: {q['question']}\nAnswer:"
        out = {"Z_closed_book": ask(SYS_Z, Q)}
        out["A_rrf"] = ask(SYS_CTX, f"Conversation excerpts:\n{ctx_text(conv, A_ids)}\n\n{Q}")
        out["B_graph"] = ask(SYS_CTX, f"Conversation excerpts:\n{ctx_text(conv, B_ids)}\n\n{Q}")
        cC = ctx_text(conv, C_ids)
        out["C_sandwich_nodraft"] = ask(SYS_CTX, f"Conversation excerpts:\n{cC}\n\n{Q}")
        out["D_sandwich_draft"] = ask(SYS_CTX, f"Conversation excerpts:\n{cC}\n\nA draft answer from an earlier pass (it may be wrong): {out['Z_closed_book']}\n\n{Q}")
        ctxs = {"A_rrf": A_ids, "B_graph": B_ids, "C_sandwich_nodraft": C_ids, "D_sandwich_draft": C_ids}
        for arm, pred in out.items():
            ids = ctxs.get(arm, [])
            rows.append(dict(qid=q["qid"], conv=sid, arm=arm, f1=score(pred, q["answer"], 1), pred=pred, gold_answer=q["answer"],
                             all_ev_in_ctx=float(gold <= set(int(i) for i in ids)) if ids else np.nan,
                             frac_ev_in_ctx=len(gold & set(int(i) for i in ids)) / len(gold) if ids else np.nan))
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(items)} calls={ncalls} {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(run.path("partial_rows.csv"), index=False)
    df = pd.DataFrame(rows); run.save_csv("per_question_arm.csv", df)
    piv = df.pivot(index="qid", columns="arm", values="f1")
    summ = [dict(arm=c, n=len(piv), **dict(zip(("mean", "lo", "hi"), st.bootstrap_ci(piv[c].values, B=10000, seed=0)))) for c in piv.columns]
    pairs = [("B_graph", "A_rrf"), ("C_sandwich_nodraft", "A_rrf"), ("D_sandwich_draft", "C_sandwich_nodraft"), ("A_rrf", "Z_closed_book")]
    pr = []
    for x, y in pairs:
        d = piv[x].values - piv[y].values; dd, lo, hi = st.bootstrap_ci(d, B=10000, seed=0)
        pr.append(dict(a=x, b=y, diff=dd, lo=lo, hi=hi, p_signflip=st.sign_flip_p(d, B=20000, seed=0), n_changed=int((d != 0).sum())))
    pr = pd.DataFrame(pr); fam = pr.iloc[:3].copy(); pr["p_holm_family3"] = st.holm(fam.p_signflip.tolist()) + [np.nan]
    run.save_csv("summary.csv", summ); run.save_csv("paired.csv", pr)
    ctxr = df[df.arm.isin(["A_rrf", "B_graph", "C_sandwich_nodraft"])].groupby("arm")[["all_ev_in_ctx", "frac_ev_in_ctx"]].mean().reset_index()
    run.save_csv("context_recall.csv", ctxr)
    run.finish(summary=dict(n=len(piv), llm_calls=ncalls, wall_s=round(time.time() - t0, 1), arms=pd.DataFrame(summ).round(4).to_dict("records"), paired=pr.round(4).to_dict("records")))
    print(pd.DataFrame(summ).round(4).to_string()); print(pr.round(4).to_string()); print(ctxr.round(3).to_string())


if __name__ == "__main__":
    main()
