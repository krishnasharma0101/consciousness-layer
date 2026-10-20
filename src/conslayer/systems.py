"""The four compared systems + sandwich variants. All share the same LLM, the same recent
window, the same answer budget and (for memory systems) the same k retrieved snippets."""
from __future__ import annotations

import time
import numpy as np

from .graph import MemoryGraph, WeightConfig, cosine
from .astar import astar_path
from .convergence import convergence_normalized, update_activation, should_refine
from .sentiment import valence

SYS = ("You are a helpful assistant with memory of a conversation with the user. "
       "Answer the user's question in one short sentence. If the user states something wrong "
       "about themselves, correct it. If you do not know, say 'I don't know'.")


def fmt_turns(turns):
    return "\n".join(f"[turn {i}] User: {t}" for i, t in turns)


def window_prompt(window, question, memories=None):
    s = ""
    if memories:
        s += "Relevant memories from earlier in the conversation:\n" + fmt_turns(memories) + "\n\n"
    s += "Recent conversation:\n" + fmt_turns(window) + f"\n\nUser question: {question}"
    return s


class Embedder:
    def __init__(self, cache_dir=None):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir, device="cpu")

    def __call__(self, texts):
        return self.m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


class Dialogue:
    """Ingests user turns once; holds embeddings, vector store and memory graph."""
    def __init__(self, turns, emb: Embedder, graph_cfg: WeightConfig, window: int):
        self.turns = list(enumerate(turns))
        self.E = emb(turns)
        self.window = self.turns[-window:]
        self.graph = MemoryGraph(graph_cfg)
        for (i, t), e in zip(self.turns, self.E):
            self.graph.add_node(t, e, t=i, valence=valence(t))


def plain(llm, d: Dialogue, q):
    return llm.chat(SYS, window_prompt(d.window, q))


def full_context(llm, d: Dialogue, q):
    return llm.chat(SYS, window_prompt(d.turns, q))


def self_refine(llm, d: Dialogue, q, draft):
    p = (window_prompt(d.window, q) + f"\n\nDraft answer: {draft}\n\nCritically check the draft "
         "against the conversation and give an improved final answer in one short sentence.")
    return llm.chat(SYS, p, max_new_tokens=60)


def rag(llm, d: Dialogue, q, emb, k):
    qe = emb([q])[0]
    sims = d.E @ qe
    top = sorted(np.argsort(-sims)[:k])
    mem = [d.turns[i] for i in top]
    return llm.chat(SYS, window_prompt(d.window, q, mem)), mem


def sandwich_retrieve(d: Dialogue, q, draft, emb, k, zeta=0.85, iters=20, seed_k=5):
    """Consciousness-layer retrieval: input activation from (question + primary answer),
    spreading activation (fixed activation rule, = personalised PageRank over Eq. 6 chain),
    A* path (Eq. 5) from best cue node to most activated node, normalised C (Eq. 7)."""
    g = d.graph
    W, order = g.adjacency()
    cue = emb([q + " " + draft])[0]
    sims = np.clip(d.E[order] @ cue, 0, None)
    I = np.zeros_like(sims)
    top = np.argsort(-sims)[:seed_k]
    I[top] = sims[top]
    if I.sum() > 0:
        I = I / I.sum()
    A = I.copy()
    for _ in range(iters):
        A = update_activation(A, W, I, zeta)
    c = convergence_normalized(A, W)
    goal = order[int(np.argmax(A))]
    start = order[int(np.argmax(sims))]
    path, _, _ = astar_path(g, start, goal) if start != goal else ([goal], 0.0, 0)
    ranked = [order[i] for i in np.argsort(-A)]
    chosen = []
    for n in [goal] + path + ranked:
        if n not in chosen:
            chosen.append(n)
        if len(chosen) >= k:
            break
    return sorted(chosen), path, c


def sandwich(llm, d: Dialogue, q, draft, emb, k, lam, direction):
    t0 = time.perf_counter()
    mem_ids, path, c = sandwich_retrieve(d, q, draft, emb, k)
    retr_time = time.perf_counter() - t0
    if not should_refine(c, lam, direction):
        return draft, retr_time, c, False, mem_ids
    mem = [d.turns[i] for i in mem_ids]
    p = (window_prompt(d.window, q, mem) + f"\n\nDraft answer (from a fast model without memory): {draft}\n\n"
         "Using the memories, give the corrected final answer in one short sentence. "
         "If a fact changed over time, use the latest turn.")
    ans, dt = llm.chat(SYS, p, max_new_tokens=60)
    # Graph update (Eq. 4). success is a label-free proxy: did the answer use a retrieved memory?
    ae = emb([ans])[0]
    success = float(max(cosine(ae, d.E[i]) for i in mem_ids) > 0.5)
    if len(path) > 1:
        d.graph.reinforce(path, success)
    return ans, retr_time + dt, c, True, mem_ids
