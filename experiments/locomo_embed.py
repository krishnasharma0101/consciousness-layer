"""Preprocessing: embed every LoCoMo turn text and question with MiniLM (CPU) and record dataset statistics.
Usage: python experiments/locomo_embed.py --run-id R-2026-11-03-01"""
import argparse, collections, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
from conslayer.runlog import Run
import locomo_common as L

ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); a = ap.parse_args()
cfg = dict(dataset="snap-research/locomo data/locomo10.json (repo commit 3eb6f2c, 2024-08-13)", dataset_sha256=L.sha256_file(L.DATA),
           license="CC BY-NC 4.0 (non-commercial, attribution) -- research use OK; data/ is not redistributed in this repo",
           embedder=L.EMBEDDER, normalize=True, node_text="'<speaker>: <text>' + ' [image: <blip_caption>]' if captioned")
run = Run(a.run_id, cfg, seed=None, models={"embedder": L.EMBEDDER})
convs = L.load_convs()
texts = {}
for c in convs:
    for t in c["texts"]: texts[L.h(t)] = t
    for q in c["qa"]: texts[L.h(q["question"])] = q["question"]
keys = sorted(texts, key=lambda k: len(texts[k]))
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(L.EMBEDDER, cache_folder=L.MODELS, device="cpu")
t0 = time.time()
E = m.encode([texts[k] for k in keys], batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
os.makedirs(L.CACHE, exist_ok=True); np.save(L.EMB, E)
import json; json.dump({k: i for i, k in enumerate(keys)}, open(L.IDX, "w"))
lens = [len(m.tokenizer(texts[k])["input_ids"]) for k in keys]
trunc = float(np.mean(np.array(lens) > m.max_seq_length))
# dataset statistics (the only things looked at before pre-registering R-02/R-03: counts, no retrieval results)
rows = []
for c in convs:
    for q in c["qa"]:
        rows.append(dict(sample_id=c["sample_id"], qid=q["qid"], category=q["category"], n_ev_raw=q["n_ev_raw"],
                         n_ev_valid=q["n_ev_valid"], dropped="|".join(q["dropped"])))
import pandas as pd
df = pd.DataFrame(rows); run.save_csv("qa_stats.csv", df)
run.save_csv("conv_stats.csv", pd.DataFrame([dict(sample_id=c["sample_id"], n_turns=len(c["texts"]), n_qa=len(c["qa"])) for c in convs]))
tab = df.assign(multi=df.n_ev_valid >= 2).groupby("category").agg(n=("qid", "size"), n_ev_ge2=("multi", "sum"),
        mean_ev=("n_ev_valid", "mean"), n_ev_gt10=("n_ev_valid", lambda s: int((s > 10).sum())), n_ev0=("n_ev_valid", lambda s: int((s == 0).sum())))
run.save_csv("category_table.csv", tab.reset_index())
run.finish(summary=dict(n_texts=len(keys), n_turns=sum(len(c["texts"]) for c in convs), n_qa=len(df), embed_s=round(time.time() - t0, 1),
                        emb_sha256=L.sha256_file(L.EMB), frac_truncated_256=trunc, n_questions_with_dropped_evidence=int((df.n_ev_valid < df.n_ev_raw).sum())
                        + 0, ))
print(tab.to_string()); print(run.meta["summary"])
