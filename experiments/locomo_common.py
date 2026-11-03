"""Shared LoCoMo (snap-research/locomo, data/locomo10.json, CC BY-NC 4.0) loading + embedding cache.
data/ is git-ignored; hashes are recorded in every run config.

Fixed preprocessing rules (pre-registered in R-2026-11-03-01):
  * node = one dialogue turn; text = "<speaker>: <text>" + (" [image: <blip_caption>]" if the turn has an image caption).
  * node time (days) = days since session_1 + turn_index/1440; session times parsed from "1:56 pm on 8 May, 2023".
  * evidence cleaning: every evidence string is scanned with  D:?(\\d+):0*(\\d+)  -> "D<a>:<b>"; ids not present in the
    conversation are dropped. Questions keep their remaining valid ids (n_ev_valid).
  * category names (by the official eval code + content check): 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
    5 adversarial.
"""
import hashlib, json, os, re
from datetime import datetime
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "locomo", "locomo10.json")
CACHE = os.path.join(ROOT, "data", "cache")
EMB = os.path.join(CACHE, "locomo_minilm.npy")
IDX = os.path.join(CACHE, "locomo_minilm_index.json")
MODELS = os.path.join(ROOT, "models")
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
CATNAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
EV = re.compile(r"D:?(\d+):0*(\d+)")


def h(t):
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def sha256_file(path, chunk=1 << 20):
    s = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            s.update(b)
    return s.hexdigest()


def parse_time(s):
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")


def clean_evidence(ev_list, valid_ids):
    out, dropped = [], []
    for e in ev_list:
        found = [f"D{a}:{b}" for a, b in EV.findall(e)]
        if not found:
            dropped.append(e)
        for f in found:
            (out if f in valid_ids else dropped).append(f)
    seen = set(); out2 = []
    for f in out:
        if f not in seen:
            seen.add(f); out2.append(f)
    return out2, dropped


def load_convs():
    """Returns list of dicts: sample_id, texts, dia_ids, times, speakers, qa=[{qid,question,answer,category,gold(set idx),n_ev_raw,n_ev_valid,dropped}]"""
    raw = json.load(open(DATA))
    convs = []
    for x in raw:
        c = x["conversation"]
        sess = sorted((int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k)))
        t0 = parse_time(c["session_1_date_time"])
        texts, dia, times, spk, dates = [], [], [], [], []
        for s in sess:
            base = (parse_time(c[f"session_{s}_date_time"]) - t0).total_seconds() / 86400
            for ti, t in enumerate(c[f"session_{s}"]):
                txt = f"{t['speaker']}: {t['text']}"
                if t.get("blip_caption"):
                    txt += f" [image: {t['blip_caption']}]"
                texts.append(txt); dia.append(t["dia_id"]); times.append(base + ti / 1440.0); spk.append(t["speaker"])
                dates.append(parse_time(c[f"session_{s}_date_time"]).strftime("%d %B %Y").lstrip("0"))
        pos = {d: i for i, d in enumerate(dia)}
        qa = []
        for qi, q in enumerate(x["qa"]):
            ev, dropped = clean_evidence(q.get("evidence", []), pos)
            qa.append(dict(qid=f"{x['sample_id']}#{qi}", question=q["question"], answer=q.get("answer", q.get("adversarial_answer")),
                           adv_answer=q.get("adversarial_answer"), category=q["category"], gold=sorted(pos[e] for e in ev),
                           n_ev_raw=len(q.get("evidence", [])), n_ev_valid=len(ev), dropped=dropped))
        convs.append(dict(sample_id=x["sample_id"], texts=texts, dia_ids=dia, times=times, speakers=spk, dates=dates, qa=qa))
    return convs


def load_cache():
    return np.load(EMB, mmap_mode="r"), json.load(open(IDX))
