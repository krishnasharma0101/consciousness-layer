"""Post-hoc (not pre-registered) analysis of R-2026-10-26-03 Part A: AUROC of -C_norm within probe type,
to separate the probe-type confound from a within-type signal. Writes results/R-2026-10-26-03/posthoc_within_ptype.csv"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from conslayer import stats as st
d = "results/R-2026-10-26-03"
A = pd.read_csv(os.path.join(d, "partA_per_probe.csv"))
rows = []
for lab in ["sandwich_always", "rag", "sandwich_nodraft"]:
    for pt, g in A.groupby("ptype"):
        y = g[f"wrong[{lab}]"].values
        for sc in ["c_norm[q+draft]", "maxcos[q+draft]"]:
            a, lo, hi = st.auroc_ci(-g[sc].values, y, B=5000, seed=0)
            rows.append(dict(label=lab, ptype=pt, score="-" + sc, n=len(y), n_pos=int(y.sum()), auroc=a, lo=lo, hi=hi))
R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "posthoc_within_ptype.csv"), index=False)
print(R.round(3).to_string())
