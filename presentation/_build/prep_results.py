"""Collects the real numbers from results/ (+ ledger) into results_data.json for results.html. Read-only on the repo."""
import json, os, re, math
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RS = os.path.join(ROOT, "results")
rd = lambda *p: pd.read_csv(os.path.join(RS, *p))
def recs(df): return json.loads(df.to_json(orient="records"))
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]
out = {}
# --- v0 synthetic
v0 = rd("v0", "summary.csv")
nd = rd("v0", "per_probe_nodraft.csv")
nd_rows = []
for pt in ("all", "recall", "update", "contradiction"):
    sub = nd if pt == "all" else nd[nd.ptype == pt]
    k, n = int(sub.correct.sum()), len(sub); lo, hi = wilson(k, n)
    nd_rows.append(dict(system="sandwich_nodraft", ptype=pt, correct=k, n=n, acc=k / n, ci_lo=lo, ci_hi=hi))
out["v0"] = dict(summary=recs(v0), nodraft=nd_rows, mcnemar=recs(rd("v0", "mcnemar.csv")),
                 retrieval_recall=recs(rd("v0", "retrieval_recall.csv")), signal=json.load(open(os.path.join(RS, "v0", "signal.json")), parse_constant=lambda x: None))
# --- E3 LongMemEval-S
s = rd("R-2026-10-26-02", "summary.csv")
out["e3"] = dict(summary=recs(s[s.metric.isin(["turn_any@10", "turn_ndcg@10", "turn_any@5", "turn_any@1"])]),
                 paired_vs_cosine=recs(rd("R-2026-10-26-02", "paired_vs_cosine.csv")),
                 primary_holm=recs(rd("R-2026-10-26-02", "primary_holm.csv")),
                 ablation_vs_v1=recs(rd("R-2026-10-26-02", "ablation_vs_v1.csv")),
                 by_qtype=recs(rd("R-2026-10-26-02", "by_qtype.csv")))
# RRF(BM25, cosine) on all 470 (from R-2026-10-26-04 per_question: first-stage rows w = 0)
pq4 = rd("R-2026-10-26-04", "per_question.csv")
base = pq4[(pq4["first"] == "rrf") & (pq4.w == 0.0)].drop_duplicates("qid")
out["e3"]["rrf_470"] = dict(n=int(len(base)), turn_any10=float(base["turn_any@10"].mean()), turn_ndcg10=float(base["turn_ndcg@10"].mean()),
                            logged_ci_turn_any10=[0.887, 0.938], logged_diff_vs_cosine=[0.057, 0.028, 0.087], logged_diff_vs_bm25=[0.055, 0.030, 0.081],
                            source="R-2026-10-26-04 per_question.csv (first=rrf, w=0); CI and diffs as logged in LAB_NOTEBOOK.md (exploratory)")
# --- E2 AUROC
out["e2"] = dict(auroc=recs(rd("R-2026-10-26-03", "auroc.csv").fillna("")), posthoc=recs(rd("R-2026-10-26-03", "posthoc_within_ptype.csv").fillna("")))
# --- E3b reranker
out["e3b"] = dict(primary=recs(rd("R-2026-10-26-04", "primary_holm.csv")), test=recs(rd("R-2026-10-26-04", "test_results.csv")),
                  grid=recs(rd("R-2026-10-26-04", "grid_means.csv")))
# --- LoCoMo
out["lc02"] = dict(summary=recs(rd("R-2026-11-03-02", "summary.csv")), primary=recs(rd("R-2026-11-03-02", "primary.csv")),
                   paired=recs(rd("R-2026-11-03-02", "paired.csv")), cluster=recs(rd("R-2026-11-03-02", "cluster_sensitivity.csv")),
                   category_table=recs(rd("R-2026-11-03-01", "category_table.csv")))
s3 = rd("R-2026-11-03-03", "summary.csv")
out["lc03"] = dict(summary=recs(s3[s3.subset.str.contains("PRIMARY")]), family=recs(rd("R-2026-11-03-03", "family_holm.csv")))
qa = rd("R-2026-11-03-04", "per_question_arm.csv")
out["lc04"] = dict(means=recs(qa.groupby("arm").f1.agg(["mean", "count"]).reset_index()),
                   logged_ci={"A_rrf": [0.222, 0.296], "B_graph": [0.222, 0.295], "C_sandwich_nodraft": [0.185, 0.257], "D_sandwich_draft": [0.194, 0.265], "Z_closed_book": [0.115, 0.179]},
                   paired=recs(rd("R-2026-11-03-04", "paired.csv").fillna("")), context_recall=recs(rd("R-2026-11-03-04", "context_recall.csv")))
# --- ledger
led = open(os.path.join(ROOT, "docs", "THEORY_LEDGER.md")).read()
rows = []
for line in led.splitlines():
    m = re.match(r"^\| ([PV]-\d+) \| (.*?) \| (.*?) \| (.*) \|\s*$", line)
    if m:
        st = m.group(3).lower()
        cat = next(c for c in ["not testable", "supported", "refuted", "mixed", "weak evidence", "untested"] if st.startswith(c))
        rows.append(dict(id=m.group(1), claim=m.group(2), status_raw=m.group(3), status=cat, evidence=m.group(4)))
out["ledger"] = rows
cnt = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
json.dump(out, open(os.path.join(os.path.dirname(__file__), "results_data.json"), "w"), separators=(",", ":"))
print(len(rows), cnt, out["e3"]["rrf_470"]["n"], round(out["e3"]["rrf_470"]["turn_any10"], 4), nd_rows[0])
print(out["lc04"]["means"])

# --- research process: run list. Times/commits from each results/<run>/config.json; prediction checks transcribed
#     item by item from LAB_NOTEBOOK.md "Prediction check" sections (no new numbers).
def cfgof(r):
    p = os.path.join(RS, r, "config.json")
    if not os.path.exists(p): return {}
    c = json.load(open(p)); return dict(start=c.get("start_ist"), wall=c.get("wall_clock_s"), commit=(c.get("git") or {}).get("short"), status=c.get("status"))
runs = [
 dict(id="R-2026-10-20-01", dir="v0", start="2026-10-20", date_note="v0 run predates the notebook (date from the run ID; time not logged); backfilled 2026-10-26", kind="backfilled",
      title="v0 synthetic: plain / self-refine / RAG / Sandwich (n=120)", verdict="Exploratory. RAG 113/120 > Sandwich 94/120; dropping the draft gives 114. Draft contamination found.", held=[], failed=[]),
 dict(id="R-2026-10-26-01", dir="R-2026-10-26-01", kind="preprocessing", title="LongMemEval-S MiniLM embedding cache (190,015 texts)", verdict="Done (1,073 s).", held=[], failed=[]),
 dict(id="R-2026-10-26-02s", dir="R-2026-10-26-02s", kind="smoke", title="E3 smoke test (--limit 10)", verdict="Checks that the code runs; numbers not interpreted.", held=[], failed=[]),
 dict(id="R-2026-10-26-02", dir="R-2026-10-26-02", kind="preregistered", title="E3: LongMemEval-S retrieval, graph vs BM25/cosine + controls/ablations (n=470)",
      verdict="V-04 refuted: v1 − cosine −0.036 [−0.062, −0.011]. Weights ≈ shuffled; topology > random.",
      held=["v1 − cosine ≈ −0.04, range [−0.10, 0] (observed −0.036)", "|BM25 − cosine| < 0.10 at @10 (+0.002)", "random topology < v1 by ≥ 0.02 (−0.062)",
            "shuffled W within ±0.02 of v1 (+0.006)", "|ΔS|, |ΔT|, |ΔE| < 0.02 at @10", "|noA* − v1| < 0.01 (−0.002)", "lower ζ moves v1 toward cosine"],
      failed=["v1_noR identical to v1: 11/470 A* paths differed (−log W is not scale-invariant), a spec error fixed in v1.1",
              "start == goal in > 50 % of questions (observed 33 %)", "recency helps knowledge-update questions (it hurt)"]),
 dict(id="R-2026-10-26-03", dir="R-2026-10-26-03", kind="preregistered", title="E2: does low C_norm predict failure? (AUROC)",
      verdict="Weak evidence. AUROC 0.63–0.68, below max-cosine for v1/cosine misses; residual 0.56–0.59. Pre-registration flaw logged: no primary label named.",
      held=["Part A AUROC ≈ 0.6, CI near 0.5 (0.671 [0.556, 0.782])", "v1 misses: AUROC in [0.45, 0.65] (0.653)", "max-cos ≥ C_norm for v1 and cosine misses",
            "residual CI includes 0.5 for v1 misses (0.564 [0.495, 0.630])"],
      failed=["cosine misses: AUROC above the range (0.679)", "max-cos ≥ C_norm for BM25 misses (0.603 < 0.625)",
              "residual CI includes 0.5 for cosine and BM25 misses (both barely exclude it)"]),
 dict(id="R-2026-10-26-04", dir="R-2026-10-26-04", kind="preregistered", title="E3b: graph as a re-ranking bonus, dev/test split (test n=247)",
      verdict="V-12 refuted: Δ 0.000 [−0.016, +0.016] (cosine), 0.000 [0, 0] (RRF). Unplanned: RRF alone 0.913 (exploratory).",
      held=["primary Δ ≤ +0.01 with CI containing 0, both first stages", "dev-chosen w is the smallest (0.1)", "shuffled W ≈ real W (|Δ| < 0.01)"], failed=[]),
 dict(id="E5/E6", dir=None, start="2026-10-26T18:07:00+05:30", kind="skipped", title="E5/E6 skipped (decision logged before running)",
      verdict="Synthetic task is saturated (120/120 recall); end-to-end QA judge needs paid API or GPU.", held=[], failed=[]),
 dict(id="R-2026-11-03-01", dir="R-2026-11-03-01", kind="preprocessing", title="LoCoMo embedding cache + dataset stats", verdict="Done.", held=[], failed=[]),
 dict(id="R-2026-11-03-02", dir="R-2026-11-03-02", kind="preregistered", title="Confirm RRF(BM25, cosine) on LoCoMo (N=1,536)",
      verdict="H1 confirmed: RRF − cosine +0.102 [0.081, 0.124]; RRF − BM25 +0.026 [0.007, 0.045] (clustered CI includes 0).",
      held=["H1 rule: both diffs > 0, both Holm p < 0.05", "RRF − BM25 in [0.00, +0.06] (+0.026)"], failed=["RRF − cosine in [+0.02, +0.08] (observed +0.102, larger)"]),
 dict(id="R-2026-11-03-03 (attempt 1)", dir="R-2026-11-03-03_ABORTED_attempt1", kind="aborted", title="Aborted: BLAS thread oversubscription (no result looked at)",
      verdict="Killed, root cause fixed (thread pinning only), re-run under the same pre-registration.", held=[], failed=[]),
 dict(id="R-2026-11-03-03", dir="R-2026-11-03-03", kind="preregistered", title="LoCoMo multi-hop: graph vs RRF, controls, ablations (N=277)",
      verdict="H2 FALSIFIED (both falsifiers fired). Best graph +0.018 [−0.004, +0.043] < +0.03 bar; nothing survives Holm.",
      held=["rr_sonly − RRF in [−0.01, +0.01] (0.000)", "rr_strong − RRF in [−0.06, +0.02] (+0.018)", "real ≈ shuffled within ±0.02 for rr_sonly and v1",
            "random topology worse than real topology", "each S/T/E ablation within ±0.02", "noR identical to full model", "H2 falsified (prior: ~7 % supported)"],
      failed=["v1 − RRF ≤ −0.05 (observed −0.025)", "real ≈ shuffled for rr_strong (+0.022, marginal)"]),
 dict(id="R-2026-11-03-04", dir="R-2026-11-03-04", kind="preregistered", title="End-to-end QA, Qwen2.5-1.5B reader, LoCoMo multi-hop F1 (N=277)",
      verdict="Graph context = RRF context: −0.0006 [−0.015, +0.013]. Falsifier for 'graph helps the reader' fired.",
      held=["A − Z ≥ +0.10 (+0.113)", "B − A in [−0.02, +0.02] (−0.0006)", "C − A < 0 (−0.038; Holm 0.11)", "A's F1 in [0.15, 0.35] (0.258)", "falsifier for 'graph helps the reader' fired"],
      failed=["D − C < 0, i.e. the draft anchors the refine pass (observed +0.009, no anchoring)"]),
]
for r in runs:
    c = cfgof(r["dir"]) if r.get("dir") else {}
    r.setdefault("start", c.get("start")); r["wall_s"] = c.get("wall") if c else None; r["commit"] = c.get("commit") if c else None
    if r["id"] == "R-2026-10-20-01": r["wall_s"] = json.load(open(os.path.join(RS, "v0", "config.json"))).get("wall_clock_s")
out["runs"] = runs
json.dump(out, open(os.path.join(os.path.dirname(__file__), "results_data.json"), "w"), separators=(",", ":"))
print("runs", len(runs), sum(len(r["held"]) for r in runs), sum(len(r["failed"]) for r in runs))
