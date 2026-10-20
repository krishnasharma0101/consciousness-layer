"""Synthetic multi-turn memory task (fixed seed). Facts stated across turns, interleaved with
chit-chat distractors, some facts later updated; probes: recall, knowledge-update, contradiction."""
from __future__ import annotations

import random

ATTRS = {
    "pet": ("my dog's name", "My dog is called {v}.", "Actually, we renamed our dog, he's now called {v}.",
            ["Biscuit", "Pickle", "Mango", "Ziggy", "Noodle", "Pepper", "Tofu", "Waffles", "Juniper", "Bramble"]),
    "city": ("the city I live in", "I live in {v}.", "Update: I just moved, I now live in {v}.",
             ["Pune", "Leeds", "Porto", "Tartu", "Ghent", "Kochi", "Bergen", "Utrecht", "Mysore", "Graz"]),
    "job": ("my job", "I work as a {v}.", "I switched careers, I now work as a {v}.",
            ["pharmacist", "glassblower", "surveyor", "beekeeper", "translator", "welder", "archivist",
             "radiographer", "sommelier", "locksmith"]),
    "sister": ("my sister's name", "My sister's name is {v}.", None,
               ["Anika", "Brielle", "Corinna", "Dalia", "Esme", "Farida", "Greta", "Halima", "Ines", "Jovana"]),
    "car": ("the colour of my car", "My car is {v}.", "I repainted my car, it's {v} now.",
            ["teal", "maroon", "beige", "mustard", "lavender", "olive", "crimson", "turquoise", "silver", "amber"]),
    "food": ("my favourite food", "My favourite food is {v}.", "These days my favourite food is {v}.",
             ["ramen", "falafel", "pierogi", "paella", "dosa", "goulash", "ceviche", "gnocchi", "bibimbap", "moussaka"]),
}
DISTRACT = [
    "The weather has been so gloomy this week, I hate it.", "I watched a great documentary about octopuses.",
    "Traffic this morning was awful, I was so stressed.", "I'm really excited about the weekend.",
    "Do you think it will rain tomorrow?", "I've been reading a book about the history of maps.",
    "My neighbour plays the drums at night, it's annoying.", "I tried a new coffee shop, it was fine.",
    "I'm feeling a bit tired today.", "I love long walks in the evening.",
    "Work has been busy but okay.", "I'm thinking of learning to play chess.",
    "I'm worried about my exam next month.", "Yesterday I cleaned the whole kitchen.",
    "Sunsets here are beautiful.", "I might take up running again.",
]


def make_dialogues(n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for di in range(n):
        attrs = rng.sample(list(ATTRS), 5)
        values = {a: rng.choice(ATTRS[a][3]) for a in attrs}
        turns = []
        facts = [ATTRS[a][1].format(v=values[a]) for a in attrs]
        dis = rng.sample(DISTRACT, 9)
        # facts early, interleaved with distractors
        for f, x in zip(facts, dis[:5]):
            turns += [f, x]
        updatable = [a for a in attrs if ATTRS[a][2]]
        upd = rng.choice(updatable)
        old = values[upd]
        new = rng.choice([v for v in ATTRS[upd][3] if v != old])
        turns.append(ATTRS[upd][2].format(v=new))
        values[upd] = new
        turns += dis[5:9]   # the recent window is always distractors
        stable = [a for a in attrs if a != upd]
        r = rng.choice(stable)
        c = rng.choice([a for a in stable if a != r])
        wrong = rng.choice([v for v in ATTRS[c][3] if v != values[c]])
        probes = [
            {"type": "recall", "q": f"What is {ATTRS[r][0]}?", "gold": values[r], "bad": None},
            {"type": "update", "q": f"What is {ATTRS[upd][0]} now?", "gold": new, "bad": old},
            {"type": "contradiction", "q": f"Just to confirm, {ATTRS[c][0]} is {wrong}, right?",
             "gold": values[c], "bad": None},
        ]
        out.append({"id": di, "turns": turns, "probes": probes})
    return out


def score(answer: str, probe: dict) -> bool:
    a = answer.lower()
    ok = probe["gold"].lower() in a
    if probe["bad"]:
        ok = ok and probe["bad"].lower() not in a
    return ok
