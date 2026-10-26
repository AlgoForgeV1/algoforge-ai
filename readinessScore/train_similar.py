# train_similar.py
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

df = pd.read_parquet("data/processed/problems.parquet").reset_index(drop=True)
slug2idx = {s: i for i, s in enumerate(df["titleSlug"])}

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000,
                        sublinear_tf=True, stop_words="english")
X_text = normalize(tfidf.fit_transform(df["text"]))
X_tags = normalize(csr_matrix(MultiLabelBinarizer().fit_transform(df["tags"])))
X_diff = normalize(csr_matrix(pd.get_dummies(df["difficulty"]).values.astype(float)))

def combine(w_text, w_tags, w_diff):
    return hstack([w_text * X_text, w_tags * X_tags, w_diff * X_diff]).tocsr()

# ground truth: LeetCode's own similar-question links (only those present in our data)
truth = {
    i: [slug2idx[s] for s in slugs if s in slug2idx]
    for i, slugs in enumerate(df["similar_slugs"])
}
truth = {i: t for i, t in truth.items() if t}

def recall_at_k(X, k=10):
    sims = (X @ X.T).toarray()
    np.fill_diagonal(sims, -1)
    top = np.argsort(-sims, axis=1)[:, :k]
    scores = [len(set(top[i]) & set(t)) / len(t) for i, t in truth.items()]
    return float(np.mean(scores))

# baseline first: text only
print("text only:", recall_at_k(combine(1, 0, 0)))

best = (0, None)
for w_text, w_tags, w_diff in itertools.product([1.0], [0, 0.3, 0.6, 1.0], [0, 0.1, 0.3]):
    r = recall_at_k(combine(w_text, w_tags, w_diff))
    print(f"text={w_text} tags={w_tags} diff={w_diff} -> recall@10={r:.3f}")
    if r > best[0]:
        best = (r, (w_text, w_tags, w_diff))
print("best:", best)

# export top-K neighbors from the full-quality similarity (no SVD)
X = combine(*best[1])
sims = (X @ X.T).toarray()
np.fill_diagonal(sims, -1)

K = 30
top = np.argsort(-sims, axis=1)[:, :K]

Path("models").mkdir(exist_ok=True)
Path("models/problem_model.json").write_text(json.dumps({
    "slugs": df["titleSlug"].tolist(),
    "titles": df["title"].tolist(),
    "difficulty": df["difficulty"].tolist(),
    "tags": [list(t) for t in df["tags"]],   # arrays -> plain lists
    "neighbors": [
        [[int(j), round(float(sims[i, j]), 3)] for j in top[i]]
        for i in range(len(df))
    ],
}))
print("Saved models/problem_model.json")