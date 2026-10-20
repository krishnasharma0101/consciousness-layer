"""v0 comparison: plain vs self-refine vs vector RAG vs Sandwich graph on the synthetic memory task.

Usage: python experiments/run_v0.py --n 40 --seed 0 --dev-n 10 --dev-seed 1 --out results/v0
"""
import argparse, copy, csv, json, os, platform, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from conslayer.graph import WeightConfig
from conslayer.llm import LocalLLM
from conslayer.synthetic import make_dialogues, score
from conslayer import systems as S

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dev-n", type=int, default=10); ap.add_argument("--dev-seed", type=int, default=1)
ap.add_argument("--window", type=int, default=4); ap.add_argument("--k", type=int, default=4)
ap.add_argument("--out", default="results/v0")
ap.add_argument("--systems", default="plain,full_context,self_refine,rag,sandwich_lambda,sandwich_always,sandwich_shuffled")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
CACHE = os.path.join(os.path.dirname(__file__), "..", "models")
cfg = WeightConfig()  # audit-fixed defaults
llm = LocalLLM(args.model, cache_dir=CACHE)
emb = S.Embedder(cache_dir=CACHE)
systems = args.systems.split(",")
for _ in range(2):
    llm.chat("You are helpful.", "Say hi.", max_new_tokens=4)  # warm-up, excluded from timings
emb(["warm up"])

# ---- 1. calibrate lambda on a disjoint dev set (different seed): lambda = median C_norm ----
dev_c = []
for dlg in make_dialogues(args.dev_n, args.dev_seed):
    d = S.Dialogue(dlg["turns"], emb, cfg, args.window)
    for p in dlg["probes"]:
        draft, _ = S.plain(llm, d, p["q"])
        _, _, c = S.sandwich_retrieve(d, p["q"], draft, emb, args.k)
        dev_c.append(c)
lam = float(np.median(dev_c))
print(f"lambda (median dev C_norm over {len(dev_c)} probes) = {lam:.5f}", flush=True)

# ---- 2. test ----
rows = []
fcsv = open(os.path.join(args.out, "per_probe.csv"), "w", newline="")
wr = csv.DictWriter(fcsv, fieldnames=["dialogue", "ptype", "system", "correct", "latency_s", "llm_calls",
                                      "c_norm", "refined", "answer", "gold"])
wr.writeheader()
t_start = time.time()
for dlg in make_dialogues(args.n, args.seed):
    base = S.Dialogue(dlg["turns"], emb, cfg, args.window)
    graphs = {"sandwich_lambda": copy.deepcopy(base), "sandwich_always": copy.deepcopy(base),
              "sandwich_shuffled": copy.deepcopy(base)}
    graphs["sandwich_shuffled"].graph.shuffle_weights(seed=args.seed + dlg["id"])
    for p in dlg["probes"]:
        q = p["q"]
        draft, t_plain = S.plain(llm, base, q)
        res = {}
        res["plain"] = (draft, t_plain, 1, None, None)
        if "full_context" in systems:
            a, t = S.full_context(llm, base, q); res["full_context"] = (a, t, 1, None, None)
        if "self_refine" in systems:
            a, t = S.self_refine(llm, base, q, draft); res["self_refine"] = (a, t_plain + t, 2, None, None)
        if "rag" in systems:
            t0 = time.perf_counter(); (a, _), _ = S.rag(llm, base, q, emb, args.k)
            res["rag"] = (a, time.perf_counter() - t0, 1, None, None)
        for name, direction, l in [("sandwich_lambda", "below", lam), ("sandwich_always", "always", lam),
                                   ("sandwich_shuffled", "always", lam)]:
            if name in systems:
                a, t, c, refined = S.sandwich(llm, graphs[name], q, draft, emb, args.k, l, direction)[:4]
                res[name] = (a, t_plain + t, 1 + int(refined), c, refined)
        for name, (a, t, calls, c, refined) in res.items():
            r = dict(dialogue=dlg["id"], ptype=p["type"], system=name, correct=int(score(a, p)),
                     latency_s=round(t, 3), llm_calls=calls, c_norm=c, refined=refined, answer=a, gold=p["gold"])
            wr.writerow(r); rows.append(r)
        fcsv.flush()
    done = dlg["id"] + 1
    print(f"dialogue {done}/{args.n}  elapsed {time.time()-t_start:.0f}s", flush=True)

meta = dict(model=args.model, embedder="sentence-transformers/all-MiniLM-L6-v2", n_dialogues=args.n,
            n_probes=3 * args.n, seed=args.seed, dev_seed=args.dev_seed, dev_n=args.dev_n, lam=lam,
            window=args.window, k=args.k, graph_cfg=cfg.__dict__, decoding="greedy, bf16, CPU",
            wall_clock_s=round(time.time() - t_start, 1), python=platform.python_version(), llm_calls=llm.calls)
json.dump(meta, open(os.path.join(args.out, "config.json"), "w"), indent=2)
print("done", json.dumps(meta)[:300])
