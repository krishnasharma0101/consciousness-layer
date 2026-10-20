"""Tiny lexicon valence in [-1,1]. Deliberately crude: E_ij's source is unspecified in the paper."""
import re

POS = set("love loved lovely happy excited great amazing wonderful enjoy enjoyed glad proud fantastic "
          "awesome fun delighted thrilled grateful beautiful best like adore".split())
NEG = set("hate hated sad angry terrible awful annoyed upset worried scared tired stressed horrible "
          "worst miss lonely frustrated afraid anxious bored".split())


def valence(text: str) -> float:
    w = re.findall(r"[a-z']+", text.lower())
    p = sum(x in POS for x in w)
    n = sum(x in NEG for x in w)
    return 0.0 if p + n == 0 else (p - n) / (p + n)
