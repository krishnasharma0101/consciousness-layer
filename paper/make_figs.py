"""Builds paper tables (LaTeX) and figures from results/ CSVs only. No numbers are typed in by hand."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="results/"; P="paper/"
PRIM="cat1_multihop_ev>=2 [PRIMARY]"
def f(x,d=3): return f"{x:.{d}f}"
def ci(m,l,h): return f"{m:.3f} [{l:.3f}, {h:.3f}]"
def sg(x,l,h): return f"{x:+.3f} [{l:+.3f}, {h:+.3f}]"

# ---- T1: R-02 retrieval on all cat1-4 + primary subset
s2=pd.read_csv(R+"R-2026-11-03-02/summary.csv"); s3=pd.read_csv(R+"R-2026-11-03-03/summary.csv")
rows=[]
for r in ["bm25","cosine","rrf"]:
    a=s2[(s2.subset=="all_cat1-4")&(s2.metric=="all_ev@10")&(s2.retriever==r)].iloc[0]
    b=s3[(s3.subset==PRIM)&(s3.metric=="all_ev@10")&(s3.retriever==r)].iloc[0]
    rows.append(f"{r.upper() if r!='cosine' else 'Cosine (MiniLM)'} & {ci(a['mean'],a['lo'],a['hi'])} & {ci(b['mean'],b['lo'],b['hi'])} ")
p2=pd.read_csv(R+"R-2026-11-03-02/paired.csv"); p2=p2[(p2.subset=="all_cat1-4")&(p2.metric=="all_ev@10")]
for c,l in [("rrf-cosine","RRF $-$ cosine"),("rrf-bm25","RRF $-$ BM25")]:
    a=p2[p2.comparison==c].iloc[0]
    fh=pd.read_csv(R+"R-2026-11-03-03/family_holm.csv"); fh=fh[fh.subset==PRIM]
    bb=fh[(fh.a=="rrf")&(fh.b==c.split('-')[1])].iloc[0]
    rows.append(f"{l} (paired) & {sg(a['diff'],a['lo'],a['hi'])} & {sg(bb['diff'],bb['lo'],bb['hi'])} ")
open(P+"tables/t1.tex","w").write(" \\\\\n".join(rows)+"\n")

# ---- T2: graph variants, primary subset
pa=pd.read_csv(R+"R-2026-11-03-03/paired_all.csv"); pa=pa[pa.subset==PRIM]
fh=pd.read_csv(R+"R-2026-11-03-03/family_holm.csv"); fh=fh[fh.subset==PRIM]
names={"rr_sonly":"RRF + S-only graph bonus","rr_strong":"RRF + full-$W$ graph bonus (strong)","v1":"Graph as first stage (v1)"}
rows=[]
for k,l in names.items():
    m=s3[(s3.subset==PRIM)&(s3.metric=="all_ev@10")&(s3.retriever==k)].iloc[0]
    d=pa[(pa.metric=="all_ev@10")&(pa.a==k)&(pa.b=="rrf")].iloc[0]
    h=fh[(fh.a==k)&(fh.b=="rrf")]; ph=f"{h.iloc[0].p_holm:.2f}" if len(h) else "--"
    rows.append(f"{l} & {ci(m['mean'],m['lo'],m['hi'])} & {sg(d['diff'],d['lo'],d['hi'])} & {ph} ")
open(P+"tables/t2.tex","w").write(" \\\\\n".join(rows)+"\n")

# ---- T3: controls and ablations (all_ev@10 and frac@10 paired diffs)
pairs=[("rr_sonly","rr_sonly_shuf","S-only: real vs shuffled $W$"),("rr_sonly","rr_sonly_rand","S-only: real vs random edges"),
("rr_strong","rr_strong_shuf","Strong: real vs shuffled $W$"),("rr_strong","rr_strong_rand","Strong: real vs random edges"),
("v1","v1_shuffledW","v1: real vs shuffled $W$"),("v1","v1_randomtopo","v1: real vs random topology"),
("v1_noS","v1","v1: drop S"),("v1_noT","v1","v1: drop T"),("v1_noE","v1","v1: drop E"),("v1_noR","v1","v1: drop R (sanity: must be 0)"),("v1_noAstar","v1","v1: drop A* path"),
("rr_strong_noS","rr_strong","Strong: drop S"),("rr_strong_noT","rr_strong","Strong: drop T"),("rr_strong_noE","rr_strong","Strong: drop E"),("rr_strong_noR","rr_strong","Strong: drop R (sanity)")]
rows=[]
for a,b,l in pairs:
    x=pa[(pa.metric=="all_ev@10")&(pa.a==a)&(pa.b==b)].iloc[0]
    y=pa[(pa.metric=="frac@10")&(pa.a==a)&(pa.b==b)].iloc[0]
    rows.append(f"{l} & {sg(x['diff'],x['lo'],x['hi'])} & {sg(y['diff'],y['lo'],y['hi'])} ")
open(P+"tables/t3.tex","w").write(" \\\\\n".join(rows)+"\n")

# ---- T4: QA
q=pd.read_csv(R+"R-2026-11-03-04/summary.csv"); qp=pd.read_csv(R+"R-2026-11-03-04/paired.csv"); cr=pd.read_csv(R+"R-2026-11-03-04/context_recall.csv").set_index("arm")
lab={"Z_closed_book":"Z: closed-book (no context)","A_rrf":"A: RRF top-10 context","B_graph":"B: graph-bonus top-10 context","C_sandwich_nodraft":"C: v1 retrieval, no draft","D_sandwich_draft":"D: v1 retrieval + closed-book draft"}
rows=[]
for a in lab:
    r=q[q.arm==a].iloc[0]; ev=f"{cr.loc[a,'all_ev_in_ctx']:.3f}" if a in cr.index else "--"
    rows.append(f"{lab[a]} & {ci(r['mean'],r['lo'],r['hi'])} & {ev} ")
open(P+"tables/t4.tex","w").write(" \\\\\n".join(rows)+"\n")
rows=[]
for _,r in qp.iterrows():
    ph=f"{r.p_holm_family3:.2f}" if pd.notna(r.p_holm_family3) else "--"
    rows.append(f"{r.a[0]} $-$ {r.b[0] if r.b[0]!='Z' else 'Z'} & {sg(r['diff'],r['lo'],r['hi'])} & {(f'{r.p_signflip:.3f}' if r.p_signflip>=0.001 else '$<$0.001')} & {ph} ")
open(P+"tables/t4b.tex","w").write(" \\\\\n".join(rows)+"\n")

# ---- Fig 1: all_ev@10 primary subset
order=["bm25","cosine","rrf","rr_sonly","rr_strong","rr_strong_shuf","rr_strong_rand","v1","v1_shuffledW","v1_randomtopo"]
ss=s3[(s3.subset==PRIM)&(s3.metric=="all_ev@10")].set_index("retriever").loc[order]
fig,ax=plt.subplots(figsize=(6.4,3.0)); x=np.arange(len(order))
ax.bar(x,ss["mean"],yerr=[ss["mean"]-ss["lo"],ss["hi"]-ss["mean"]],capsize=2,color="0.65")
ax.set_xticks(x); ax.set_xticklabels(order,rotation=40,ha="right",fontsize=8); ax.set_ylabel("all-evidence@10"); ax.set_ylim(0,0.2)
ax.set_title("LoCoMo multi-hop (N=277), 95% bootstrap CI",fontsize=9); plt.tight_layout(); plt.savefig(P+"figs/fig_retrieval.pdf"); plt.close()
# ---- Fig 2: QA F1
qq=q.set_index("arm").loc[list(lab)]
fig,ax=plt.subplots(figsize=(6.0,2.7)); x=np.arange(len(qq))
ax.bar(x,qq["mean"],yerr=[qq["mean"]-qq["lo"],qq["hi"]-qq["mean"]],capsize=2,color="0.65")
ax.set_xticks(x); ax.set_xticklabels(["Z closed-book","A RRF","B graph","C no-draft","D draft"],fontsize=8); ax.set_ylabel("partial-F1")
ax.set_title("LoCoMo multi-hop QA, Qwen2.5-1.5B (N=277)",fontsize=9); plt.tight_layout(); plt.savefig(P+"figs/fig_qa.pdf"); plt.close()
# ---- Fig 3: paired diffs forest vs RRF / vs shuffled on primary metric
sel=[("rr_sonly","rrf"),("rr_strong","rrf"),("v1","rrf"),("rr_strong","rr_strong_shuf"),("rr_strong","rr_strong_rand"),("v1","v1_shuffledW"),("v1","v1_randomtopo"),("rrf","bm25"),("rrf","cosine")]
d=[pa[(pa.metric=="all_ev@10")&(pa.a==a)&(pa.b==b)].iloc[0] for a,b in sel]
fig,ax=plt.subplots(figsize=(6.0,3.0)); y=np.arange(len(sel))[::-1]
ax.errorbar([r['diff'] for r in d],y,xerr=[[r['diff']-r['lo'] for r in d],[r['hi']-r['diff'] for r in d]],fmt="o",color="k",capsize=2)
ax.axvline(0,color="0.5",lw=0.8); ax.axvline(0.03,color="r",lw=0.8,ls="--"); ax.text(0.031,y[0]+0.35,"+0.03 margin",color="r",fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([f"{a} $-$ {b}".replace("_","\\_") for a,b in sel],fontsize=7); ax.set_xlabel("paired difference in all-evidence@10 (95% CI)")
plt.tight_layout(); plt.savefig(P+"figs/fig_paired.pdf"); plt.close()
print("ok")
