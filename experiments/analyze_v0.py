"""Summarise results/v0/per_probe.csv -> summary.csv, mcnemar.csv, accuracy.png, auroc."""
import json, math, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "results/v0"
df = pd.read_csv(os.path.join(d, "per_probe.csv"))
ORDER = ["plain", "self_refine", "rag", "sandwich_lambda", "sandwich_always", "sandwich_shuffled", "full_context"]
ORDER = [s for s in ORDER if s in df.system.unique()]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)

rows = []
for s in ORDER:
    g = df[df.system == s]
    for pt in ["all", "recall", "update", "contradiction"]:
        h = g if pt == "all" else g[g.ptype == pt]
        k, n = int(h.correct.sum()), len(h)
        lo, hi = wilson(k, n)
        rows.append(dict(system=s, ptype=pt, correct=k, n=n, acc=round(k / n, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                         latency_mean_s=round(h.latency_s.mean(), 2), latency_median_s=round(h.latency_s.median(), 2),
                         llm_calls_mean=round(h.llm_calls.mean(), 2)))
summ = pd.DataFrame(rows); summ.to_csv(os.path.join(d, "summary.csv"), index=False)
print(summ[summ.ptype == "all"].to_string(index=False))
print(summ.pivot(index="system", columns="ptype", values="correct").loc[ORDER].to_string())

# McNemar exact (two-sided) vs rag and vs plain
def mcnemar(a, b):
    b01 = int(((a == 0) & (b == 1)).sum()); b10 = int(((a == 1) & (b == 0)).sum()); n = b01 + b10
    if n == 0: return b01, b10, 1.0
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b01, b10, p
piv = df.pivot_table(index=["dialogue", "ptype"], columns="system", values="correct")
mc = []
for ref in ["rag", "plain"]:
    for s in ORDER:
        if s == ref: continue
        b01, b10, p = mcnemar(piv[ref].values, piv[s].values)
        mc.append(dict(ref=ref, system=s, sys_only_correct=b01, ref_only_correct=b10, p_exact=round(p, 4)))
mc = pd.DataFrame(mc); mc.to_csv(os.path.join(d, "mcnemar.csv"), index=False); print(mc.to_string(index=False))

# AUROC: does low C_norm predict a wrong plain draft?
def auroc(score, label):  # P(score_pos > score_neg)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))
sa = df[df.system == "sandwich_always"].set_index(["dialogue", "ptype"])
pl = df[df.system == "plain"].set_index(["dialogue", "ptype"])
au_plain = auroc(-sa.c_norm.values, 1 - pl.loc[sa.index].correct.values)
au_sand = auroc(-sa.c_norm.values, 1 - sa.correct.values)
lam_rows = df[df.system == "sandwich_lambda"]
extra = dict(auroc_lowC_predicts_plain_wrong=au_plain, auroc_lowC_predicts_sandwich_wrong=au_sand,
             n_plain_wrong=int((1 - pl.correct).sum()), refine_rate_lambda=float(lam_rows.refined.astype(str).eq("True").mean()))
json.dump(extra, open(os.path.join(d, "signal.json"), "w"), indent=2); print(extra)

# plain chart
a = summ[summ.ptype != "all"].pivot(index="system", columns="ptype", values="acc").loc[ORDER][["recall", "update", "contradiction"]]
lo = summ[summ.ptype != "all"].pivot(index="system", columns="ptype", values="ci_lo").loc[ORDER][a.columns]
hi = summ[summ.ptype != "all"].pivot(index="system", columns="ptype", values="ci_hi").loc[ORDER][a.columns]
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(ORDER)); w = 0.27
for i, c in enumerate(a.columns):
    ax.bar(x + (i - 1) * w, a[c], w, yerr=[a[c] - lo[c], hi[c] - a[c]], capsize=2, label=c)
ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=20, ha="right"); ax.set_ylim(0, 1.05)
n = int(summ[(summ.ptype == "recall")].n.iloc[0])
ax.set_ylabel("accuracy"); ax.set_title(f"v0 synthetic memory task, n={n} probes per type, 95% Wilson CI")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(d, "accuracy.png"), dpi=120)
