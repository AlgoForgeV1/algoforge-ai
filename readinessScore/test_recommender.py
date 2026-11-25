# test_lgbm.py
"""
Evaluate a trained LightGBM re-ranker against the baseline.
Compares: Recall@K, NDCG@K, MAP@K for LightGBM vs raw SBERT cosine.
Run after train_lgbm.py has produced models/lgbm_ranker.txt.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_DIR = Path("models")
PARQUET_PATH = "data/processed/problems.parquet"
SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_K = 100

# ─────────────────────────── load artefacts ───────────────────────────
df = pd.read_parquet(PARQUET_PATH).reset_index(drop=True)
n_problems = len(df)
slug_to_idx = {s: i for i, s in enumerate(df["titleSlug"])}

truth = {
    i: {slug_to_idx[s] for s in slugs if s in slug_to_idx and s != df["titleSlug"][i]}
    for i, slugs in enumerate(df["similar_slugs"])
}
truth = {i: t for i, t in truth.items() if t}
print(f"Problems with ground truth: {len(truth)}")

booster_path = MODEL_DIR / "lgbm_ranker.txt"
if not booster_path.exists():
    raise FileNotFoundError(f"{booster_path} not found — run train_lgbm.py first")

ranker = lgb.Booster(model_file=str(booster_path))
print(f"Loaded LightGBM booster from {booster_path}")

feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
print(f"Features: {feature_names}")

# ─────────────────────────── embeddings ───────────────────────────
print(f"Encoding with {SBERT_NAME}...")
encoder = SentenceTransformer(SBERT_NAME)
X_emb = encoder.encode(
    df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

diff_dummies = pd.get_dummies(df["difficulty"])
diff_matrix = diff_dummies.values.astype(np.float32)

# ─────────────────────────── features (must match training) ───────────────────────────
def pair_features(i: int, j: int) -> dict:
    emb_cos = float(X_emb[i] @ X_emb[j])
    tags_i = set(df["tags"][i]); tags_j = set(df["tags"][j])
    union = tags_i | tags_j
    tag_jaccard = len(tags_i & tags_j) / len(union) if union else 0.0
    tag_overlap_count = len(tags_i & tags_j)
    diff_i = int(np.argmax(diff_matrix[i])); diff_j = int(np.argmax(diff_matrix[j]))
    diff_match = 1.0 if diff_i == diff_j else (0.5 if abs(diff_i - diff_j) == 1 else 0.0)
    len_i = len(df["text"][i]); len_j = len(df["text"][j])
    text_len_diff = abs(len_i - len_j) / max(len_i, len_j, 1)
    text_len_ratio = min(len_i, len_j) / max(len_i, len_j, 1)
    hints_i = int(df["n_hints"][i]); hints_j = int(df["n_hints"][j])
    hints_diff = abs(hints_i - hints_j)
    hints_ratio = min(hints_i, hints_j) / max(hints_i, hints_j, 1) if max(hints_i, hints_j) > 0 else 1.0
    return {
        "emb_cos": emb_cos,
        "tag_jaccard": tag_jaccard,
        "tag_overlap_count": tag_overlap_count,
        "diff_match": diff_match,
        "text_len_diff": text_len_diff,
        "text_len_ratio": text_len_ratio,
        "hints_diff": hints_diff,
        "hints_ratio": hints_ratio,
    }

# ─────────────────────────── evaluate ───────────────────────────
def evaluate(mode: str, k_values=(5, 10, 30)):
    """
    mode = 'lgbm'      → re-rank candidates with LightGBM
    mode = 'baseline'  → raw SBERT cosine (no re-ranking)
    """
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    maps = {k: [] for k in k_values}

    for i in truth:
        # Candidate generation (same for both modes)
        sims = X_emb[i] @ X_emb.T
        sims[i] = -1.0
        cands = np.argsort(-sims)[:CANDIDATE_K]

        if mode == "baseline":
            scores = sims[cands]
        else:
            feats = np.asarray([[pair_features(i, int(j))[f] for f in feature_names] for j in cands], dtype=np.float32)
            scores = ranker.predict(feats)

        order = np.argsort(-scores)
        ranked = cands[order]
        ranked_scores = scores[order]
        truth_set = truth[i]

        for k in k_values:
            top_k = ranked[:k]
            hits = len(set(top_k.tolist()) & truth_set)
            recalls[k].append(hits / len(truth_set))

            # NDCG@K (binary relevance)
            dcg = 0.0
            idcg = 0.0
            for rank, idx in enumerate(top_k):
                rel = 1.0 if idx in truth_set else 0.0
                dcg += rel / np.log2(rank + 2)
            for rank in range(min(k, len(truth_set))):
                idcg += 1.0 / np.log2(rank + 2)
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

            # MAP@K (binary)
            hit_count = 0
            precision_sum = 0.0
            for rank, idx in enumerate(top_k):
                if idx in truth_set:
                    hit_count += 1
                    precision_sum += hit_count / (rank + 1)
            maps[k].append(precision_sum / len(truth_set))

    return (
        {k: float(np.mean(v)) for k, v in recalls.items()},
        {k: float(np.mean(v)) for k, v in ndcgs.items()},
        {k: float(np.mean(v)) for k, v in maps.items()},
    )

print("\nEvaluating baseline (SBERT cosine)...")
base_rec, base_ndcg, base_map = evaluate("baseline")

print("Evaluating LightGBM re-ranker...")
lgbm_rec, lgbm_ndcg, lgbm_map = evaluate("lgbm")

# ─────────────────────────── report ───────────────────────────
print(f"\n{'='*72}")
print(f"{'Metric':<18} {'Baseline':>12} {'LightGBM':>12} {'Δ':>12}")
print(f"{'='*72}")
for k in (5, 10, 30):
    print(f"{f'Recall@{k}':<18} {base_rec[k]:>12.4f} {lgbm_rec[k]:>12.4f} {lgbm_rec[k]-base_rec[k]:>+12.4f}")
print()
for k in (5, 10, 30):
    print(f"{f'NDCG@{k}':<18} {base_ndcg[k]:>12.4f} {lgbm_ndcg[k]:>12.4f} {lgbm_ndcg[k]-base_ndcg[k]:>+12.4f}")
print()
for k in (5, 10, 30):
    print(f"{f'MAP@{k}':<18} {base_map[k]:>12.4f} {lgbm_map[k]:>12.4f} {lgbm_map[k]-base_map[k]:>+12.4f}")
print(f"{'='*72}")

# Sanity check: top-5 for a few known problems
print("\nSpot check — top 5 neighbors for known problems:")
for slug in ["two-sum", "3sum", "trapping-rain-water"]:
    if slug not in slug_to_idx:
        continue
    i = slug_to_idx[slug]
    sims = X_emb[i] @ X_emb.T
    sims[i] = -1
    cands = np.argsort(-sims)[:CANDIDATE_K]
    feats = np.asarray([[pair_features(i, int(j))[f] for f in feature_names] for j in cands], dtype=np.float32)
    scores = ranker.predict(feats)
    order = np.argsort(-scores)[:5]
    top5 = cands[order]
    print(f"\n  {slug} ({df['difficulty'][i]})")
    print(f"    ground truth similar: {sorted(truth.get(i, set()))[:5]}")
    for j in top5:
        gt_flag = "✅" if j in truth.get(i, set()) else "  "
        print(f"    {gt_flag} {df['titleSlug'][j]:<45} {df['difficulty'][j]:<8} score={scores[order][list(top5).index(j)]:.4f}")