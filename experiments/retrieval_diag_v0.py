"""LLM-free diagnostic: does the retrieved memory set contain the gold fact (and for update probes,
the latest value)? Uses the plain drafts logged in per_probe.csv as the sandwich cue. Graphs are
rebuilt fresh (no R updates), so this isolates retrieval quality."""
import copy, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from conslayer.graph import WeightConfig
from conslayer.synthetic import make_dialogues
from conslayer import systems as S

d = sys.argv[1] if len(sys.argv) > 1 else "results/v0"
df = pd.read_csv(os.path.join(d, "per_probe.csv"))
draft = df[df.system == "plain"].set_index(["dialogue", "ptype"]).answer
emb = S.Embedder(cache_dir=os.path.join(os.path.dirname(__file__), "..", "models"))
k = 4
configs = {"sandwich(default)": WeightConfig(), "sandwich_shuffled": WeightConfig(),
           "sandwich_noT": WeightConfig(use_T=False), "sandwich_noE": WeightConfig(use_E=False),
           "sandwich_onlyS": WeightConfig(use_T=False, use_E=False, use_R=False)}
rows = []
for dlg in make_dialogues(40, 0):
    base = {name: S.Dialogue(dlg["turns"], emb, cfg, 4) for name, cfg in configs.items()}
    base["sandwich_shuffled"].graph.shuffle_weights(seed=dlg["id"])
    for p in dlg["probes"]:
        q, gold = p["q"], p["gold"].lower()
        dr = str(draft.loc[(dlg["id"], p["type"])])
        (_, mem) = (None, None)
        qe = emb([q])[0]; sims = base["sandwich(default)"].E @ qe
        rag_ids = sorted(np.argsort(-sims)[:k])
        sets = {"rag": rag_ids}
        for name in configs:
            ids, _, _ = S.sandwich_retrieve(base[name], q, dr, emb, k)
            sets[name] = ids
        for name, ids in sets.items():
            txt = " ".join(dlg["turns"][i].lower() for i in ids)
            rows.append(dict(dialogue=dlg["id"], ptype=p["type"], retriever=name, gold_in_memory=int(gold in txt)))
r = pd.DataFrame(rows)
out = r.pivot_table(index="retriever", columns="ptype", values="gold_in_memory", aggfunc="sum")
out["all"] = out.sum(axis=1)
out.to_csv(os.path.join(d, "retrieval_recall.csv"))
print("gold value present in the k=4 retrieved turns (counts out of 40 per type, 120 total):")
print(out.to_string())
