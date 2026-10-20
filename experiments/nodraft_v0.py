"""Follow-up: sandwich retrieval but the secondary prompt WITHOUT the primary draft (i.e. the exact
RAG prompt). Isolates draft anchoring from retrieval. Same seed/model/k/window as v0."""
import os, sys, time, csv
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from conslayer.graph import WeightConfig
from conslayer.llm import LocalLLM
from conslayer.synthetic import make_dialogues, score
from conslayer import systems as S
d = "results/v0"; C = os.path.join(os.path.dirname(__file__), "..", "models")
draft = pd.read_csv(f"{d}/per_probe.csv").query("system=='plain'").set_index(["dialogue", "ptype"]).answer
llm = LocalLLM("Qwen/Qwen2.5-1.5B-Instruct", cache_dir=C); emb = S.Embedder(cache_dir=C)
llm.chat("You are helpful.", "Say hi.", max_new_tokens=4)
out = open(f"{d}/per_probe_nodraft.csv", "w", newline=""); w = csv.writer(out)
w.writerow(["dialogue", "ptype", "system", "correct", "latency_s", "answer", "gold"])
for dlg in make_dialogues(40, 0):
    dd = S.Dialogue(dlg["turns"], emb, WeightConfig(), 4)
    for p in dlg["probes"]:
        t0 = time.perf_counter()
        ids, _, _ = S.sandwich_retrieve(dd, p["q"], str(draft.loc[(dlg["id"], p["type"])]), emb, 4)
        a, _ = llm.chat(S.SYS, S.window_prompt(dd.window, p["q"], [dd.turns[i] for i in ids]))
        w.writerow([dlg["id"], p["type"], "sandwich_nodraft", int(score(a, p)), round(time.perf_counter() - t0, 3), a, p["gold"]])
out.close()
r = pd.read_csv(f"{d}/per_probe_nodraft.csv")
print(r.groupby("ptype").correct.sum().to_dict(), "total", r.correct.sum(), "/", len(r), "mean latency (excl. primary)", round(r.latency_s.mean(), 2))
