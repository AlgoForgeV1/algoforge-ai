import json
from collections import defaultdict

d = json.load(open("models/problem_model.json"))
slugs, titles, diffs, tags, neighbors = (
    d["slugs"], d["titles"], d["difficulty"], d["tags"], d["neighbors"]
)
idx = {s: i for i, s in enumerate(slugs)}

# newest first: (slug, status)
history = [
    ("3sum", "Wrong Answer"),
    ("two-sum", "Accepted"),
    ("valid-parentheses", "Accepted"),
    ("container-with-most-water", "Time Limit Exceeded"),
]

def recommend(history, n=8, decay=0.85):
    seen = {s for s, _ in history}
    scores = defaultdict(float)
    for rank, (slug, status) in enumerate(history):
        if slug not in idx:
            continue
        w = decay ** rank * (1.0 if status == "Accepted" else 1.5)
        for j, sim in neighbors[idx[slug]]:
            scores[j] += w * sim

    order = {"Easy": 0, "Medium": 1, "Hard": 2}
    solved = [order[diffs[idx[s]]] for s, st in history if st == "Accepted" and s in idx]
    ceiling = (max(solved) + 1) if solved else 1

    ranked = sorted(
        (j for j in scores if slugs[j] not in seen and order[diffs[j]] <= ceiling),
        key=lambda j: -scores[j],
    )
    return ranked[:n]

for j in recommend(history):
    print(f"{slugs[j]:45} {diffs[j]:7} {tags[j]}")