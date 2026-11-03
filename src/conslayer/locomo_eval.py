"""LoCoMo answer scoring: re-implementation of the official partial-F1 (snap-research/locomo task_eval/evaluation.py,
CC BY-NC 4.0 data/code) without its bert_score/rouge imports. Same normalisation (strip commas, lowercase, drop punctuation,
drop articles a/an/the/and, Porter stemming). Category 1 uses the multi-answer F1 (split on ','); 2,3,4 use token F1;
category 3 uses the part of the gold answer before ';'."""
import re, string
from collections import Counter
import numpy as np
from nltk.stem import PorterStemmer

_ps = PorterStemmer()


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    p = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    g = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def f1_multi(prediction: str, ground_truth: str) -> float:
    preds = [x.strip() for x in prediction.split(",")]
    gts = [x.strip() for x in ground_truth.split(",")]
    return float(np.mean([max(f1_score(p, g) for p in preds) for g in gts]))


def score(prediction: str, answer, category: int) -> float:
    answer = str(answer)
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        return f1_multi(prediction, answer)
    if category in (2, 3, 4):
        return f1_score(prediction, answer)
    raise ValueError("category 5 uses a different rule (not implemented; not used here)")
