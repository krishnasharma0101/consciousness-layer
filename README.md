# consciousness-layer

Experiments for Krishna Sharma's paper *"A Mathematical Framework for Artificial Consciousness:
Graph-Theoretic Mediation of Logic and Emotion in Large Language Models"* (`paper.pdf`).

**Scope, bluntly.** This repo tests the paper's graph layer as an **LLM memory / consistency
mechanism**. It does not and cannot test consciousness: no experiment here can tell a conscious system
from a non-conscious one computing the same thing. Φ and C(t) are engineering quantities.

## Layout
```
paper.pdf, paper.txt         the paper (+ pdftotext -layout)
docs/MATH_AUDIT.md           20 gaps/inconsistencies in Eqs. (1)-(10), with fixes and tests
docs/EXPERIMENT_PLAN.md      prioritised falsifiable experiments, datasets, compute
EXPERIMENTS.md               run log (config, seed, results)
src/conslayer/
  graph.py        W_ij = S*T*(1+E)*R  (Eq. 1-4), every term toggleable, audit fixes as options, tau (Eq. 9)
  astar.py        A* with cost 1/(W+eps) or -log W; admissible blended heuristic (Eq. 5)
  random_walk.py  P(i->j) = W_ij / sum W_ik with dangling/restart fixes (Eq. 6)
  convergence.py  C(t) (Eq. 7), normalised C, Phi paper/dynamic/min-cut (Eq. 8), activation rule, lambda trigger (Eq. 10)
  systems.py      plain, self-refine, vector RAG, full-context, Sandwich (primary -> graph -> secondary + graph update)
  synthetic.py    synthetic multi-turn memory task (recall / update / contradiction probes)
  llm.py          local HF transformers wrapper (CPU, greedy)
  sentiment.py    tiny lexicon valence for E_ij
tests/test_core.py           unit tests for the maths
experiments/run_v0.py        v0 comparison;  experiments/analyze_v0.py -> summary/CSV/plot
results/v0/                  per_probe.csv, summary.csv, mcnemar.csv, signal.json, accuracy.png, config.json
```

## Setup (CPU)
```
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python experiments/run_v0.py --n 40 --seed 0 --out results/v0   # downloads Qwen2.5-1.5B-Instruct + MiniLM to models/
.venv/bin/python experiments/analyze_v0.py results/v0
```
Models are downloaded into `models/` (git-ignored). No paid APIs are used.
