"""Shared LongMemEval loading + embedding cache (data/ is git-ignored; hashes recorded in configs)."""
import hashlib, json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "longmemeval_s_cleaned.json")
CACHE = os.path.join(ROOT, "data", "cache")
EMB = os.path.join(CACHE, "lme_s_minilm.npy")
IDX = os.path.join(CACHE, "lme_s_minilm_index.json")
MODELS = os.path.join(ROOT, "models")
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_file(path, chunk=1 << 20):
    s = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            s.update(b)
    return s.hexdigest()


def load():
    return json.load(open(DATA))


def load_cache():
    E = np.load(EMB, mmap_mode="r")
    idx = json.load(open(IDX))
    return E, idx
