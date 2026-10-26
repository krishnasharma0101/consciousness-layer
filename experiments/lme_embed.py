"""Preprocessing: embed every unique LongMemEval-S turn text and question with MiniLM (CPU).
Usage: python experiments/lme_embed.py --run-id R-2026-10-26-01"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from conslayer.runlog import Run
import lme_common as L

ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args()
cfg = dict(dataset="xiaowu0162/longmemeval-cleaned longmemeval_s_cleaned.json", dataset_sha256=L.sha256_file(L.DATA),
           embedder=L.EMBEDDER, normalize=True, batch=a.batch, max_seq_length=256, device="cpu")
run = Run(a.run_id, cfg, seed=None, models={"embedder": L.EMBEDDER})
d = L.load()
texts = {}
for x in d:
    texts[L.h(x["question"])] = x["question"]
    for s in x["haystack_sessions"]:
        for t in s:
            texts[L.h(t["content"])] = t["content"]
keys = sorted(texts, key=lambda k: len(texts[k]))
run.note(f"{len(keys)} unique texts (turns + questions)")
from sentence_transformers import SentenceTransformer
import torch
m = SentenceTransformer(L.EMBEDDER, cache_folder=L.MODELS, device="cpu")
run.meta["config"]["max_seq_length_actual"] = m.max_seq_length; run.meta["config"]["torch_threads"] = torch.get_num_threads()
t0 = time.time()
E = np.zeros((len(keys), 384), np.float32)
B = 4096
for s in range(0, len(keys), B):
    E[s:s + B] = m.encode([texts[k] for k in keys[s:s + B]], batch_size=a.batch, normalize_embeddings=True, show_progress_bar=False)
    print(f"{s + B}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
os.makedirs(L.CACHE, exist_ok=True)
np.save(L.EMB, E)
import json
json.dump({k: i for i, k in enumerate(keys)}, open(L.IDX, "w"))
tok = m.tokenizer
lens = [len(tok(texts[k], add_special_tokens=True)["input_ids"]) for k in keys[::50]]
trunc = float(np.mean(np.array(lens) > m.max_seq_length))
run.finish(summary=dict(n_texts=len(keys), embed_s=round(time.time() - t0, 1), emb_sha256=L.sha256_file(L.EMB),
                        frac_truncated_sample=trunc, n_sampled_for_trunc=len(lens)))
print("done", run.meta["summary"])
