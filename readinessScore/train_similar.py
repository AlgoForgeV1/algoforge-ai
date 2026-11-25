# train_lgbm.py
"""
Train a LightGBM re-ranker for problem similarity.
Two-stage pipeline:
  1. SBERT embeddings → candidate generation (top 100 per problem)
  2. LightGBM LambdaRank → re-rank candidates using pairwise features
Output: models/problem_model.json  (same schema as train_similar.py)
"""
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.model_selection import GroupShuffleSplit

# ─────────────────────────── config ───────────────────────────
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

PARQUET_PATH = "data/processed/problems.parquet"
SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_K = 100        # candidates per problem (retrieval stage)
EXPORT_K = 30            # final neighbors per problem (export stage)
RANDOM_SEED = 42

# ─────────────────────────── load data ───────────────────────────
df = pd.read_parquet(PARQUET_PATH).reset_index(drop=True)
n_problems = len(df)
print(f"Loaded {n_problems} problems")

slug_to_idx = {s: i for i, s in enumerate(df["titleSlug"])}

# Ground truth: LeetCode's similarQuestions, restricted to our data
truth = {
    i: {slug_to_idx[s] for s in slugs if s in slug_to_idx and s != df["titleSlug"][i]}
    for i, slugs in enumerate(df["similar_slugs"])
}
truth = {i: t for i, t in truth.items() if t}
print(f"Problems with ground-truth similar links: {len(truth)}")

# ─────────────────────────── embeddings ───────────────────────────
print(f"Encoding {n_problems} problems with {SBERT_NAME}...")
encoder = SentenceTransformer(SBERT_NAME)
X_emb = encoder.encode(
    df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)          # (n, 384)

# Tag and difficulty one-hot (for features, not for retrieval)
mlb = MultiLabelBinarizer()
X_tags_bin = mlb.fit_transform(df["tags"])           # (n, n_tags)
diff_dummies = pd.get_dummies(df["difficulty"])
diff_matrix = diff_dummies.values.astype(np.float32)

# ─────────────────────────── candidate generation ───────────────────────────
print(f"Generating top-{CANDIDATE_K} candidates per problem via cosine...")
# Chunked to avoid a 4000×4000 float64 matrix (approx 128 MB — fine, but chunking is safer for larger data)
candidates = {}   # problem_idx -> np.array of candidate indices
for i in range(n_problems):
    sims = X_emb[i] @ X_emb.T
    sims[i] = -1.0
    candidates[i] = np.argsort(-sims)[:CANDIDATE_K]

# ─────────────────────────── pairwise features ───────────────────────────
def pair_features(i: int, j: int) -> dict:
    """Features for a single (query_problem, candidate) pair."""
    emb_cos = float(X_emb[i] @ X_emb[j])

    tags_i = set(df["tags"][i])
    tags_j = set(df["tags"][j])
    union = tags_i | tags_j
    tag_jaccard = len(tags_i & tags_j) / len(union) if union else 0.0
    tag_overlap_count = len(tags_i & tags_j)

    diff_i = int(np.argmax(diff_matrix[i]))
    diff_j = int(np.argmax(diff_matrix[j]))
    diff_match = 1.0 if diff_i == diff_j else (0.5 if abs(diff_i - diff_j) == 1 else 0.0)

    len_i = len(df["text"][i])
    len_j = len(df["text"][j])
    text_len_diff = abs(len_i - len_j) / max(len_i, len_j, 1)
    text_len_ratio = min(len_i, len_j) / max(len_i, len_j, 1)

    hints_i = int(df["n_hints"][i])
    hints_j = int(df["n_hints"][j])
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

FEATURE_NAMES = list(pair_features(0, 1).keys())
print(f"Feature columns ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

# Build the training matrix
print("Building pairwise feature matrix...")
rows = []
labels = []
groups = []
query_ids = []

for i in range(n_problems):
    cands = candidates[i]
    group_size = 0
    for j in cands:
        j = int(j)
        feats = pair_features(i, j)
        label = 1 if (i in truth and j in truth[i]) else 0
        rows.append([feats[k] for k in FEATURE_NAMES])
        labels.append(label)
        query_ids.append(i)
        group_size += 1
    groups.append(group_size)

X_train = np.asarray(rows, dtype=np.float32)
y_train = np.asarray(labels, dtype=np.int32)
groups = np.asarray(groups, dtype=np.int32)

print(f"Pairwise matrix: {X_train.shape}, positives: {y_train.sum():,} ({100*y_train.mean():.2f}%)")
print(f"Groups (one per problem): {len(groups)}, rows per group: {groups[0]}")

# ─────────────────────────── train/validation split ───────────────────────────
# Group split: hold out 15% of problems entirely (not just rows) so eval is honest
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
train_idx, val_idx = next(splitter.split(X_train, y_train, groups=query_ids))

X_tr, X_va = X_train[train_idx], X_train[val_idx]
y_tr, y_va = y_train[train_idx], y_train[val_idx]

# Regenerate groups for the split
def rebuild_groups(indices, original_groups, query_ids_arr):
    qids = [query_ids_arr[k] for k in indices]
    new_groups = []
    prev = None
    count = 0
    for q in qids:
        if q != prev:
            if prev is not None:
                new_groups.append(count)
            prev = q
            count = 1
        else:
            count += 1
    if count > 0:
        new_groups.append(count)
    return np.asarray(new_groups, dtype=np.int32)

# Simpler: since we built rows grouped by problem, we can compute group sizes directly
train_qids = [query_ids[k] for k in train_idx]
val_qids = [query_ids[k] for k in val_idx]

from collections import Counter
train_counts = Counter(train_qids)
val_counts = Counter(val_qids)
groups_tr = np.asarray([train_counts[q] for q in sorted(train_counts)], dtype=np.int32)
groups_va = np.asarray([val_counts[q] for q in sorted(val_counts)], dtype=np.int32)

print(f"Train: {X_tr.shape[0]:,} rows across {len(groups_tr)} problems")
print(f"Val:   {X_va.shape[0]:,} rows across {len(groups_va)} problems")

# ─────────────────────────── train ───────────────────────────
print("\nTraining LightGBM LambdaRank...")
ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=-1,
)

ranker.fit(
    X_tr, y_tr,
    group=groups_tr,
    eval_set=[(X_va, y_va)],
    eval_group=[groups_va],
    eval_at=[5, 10],
)

# ─────────────────────────── evaluate on validation ───────────────────────────
def recall_at_k_from_ranker(ranker, X, y, groups, k=10):
    """Compute Recall@K on grouped validation data."""
    preds = ranker.predict(X)
    offset = 0
    recalls = []
    for g in groups:
        p = preds[offset:offset + g]
        t = y[offset:offset + g]
        offset += g
        if t.sum() == 0:
            continue
        top_k_idx = np.argsort(-p)[:k]
        hits = t[top_k_idx].sum()
        recalls.append(hits / t.sum())
    return float(np.mean(recalls)) if recalls else 0.0

val_recall10 = recall_at_k_from_ranker(ranker, X_va, y_va, groups_va, k=10)
print(f"\nValidation Recall@10 (LightGBM): {val_recall10:.4f}")

# Baseline: pure embedding cosine
def baseline_recall_at_k(X_emb, val_qids_set, truth, k=10):
    recalls = []
    for i in val_qids_set:
        if i not in truth:
            continue
        sims = X_emb[i] @ X_emb.T
        sims[i] = -1
        top_k = set(np.argsort(-sims)[:k].tolist())
        recalls.append(len(top_k & truth[i]) / len(truth[i]))
    return float(np.mean(recalls)) if recalls else 0.0

val_problem_set = set(val_qids)
baseline_recall10 = baseline_recall_at_k(X_emb, val_problem_set, truth, k=10)
print(f"Validation Recall@10 (SBERT cosine baseline): {baseline_recall10:.4f}")
print(f"Improvement: {(val_recall10 - baseline_recall10) * 100:+.2f} percentage points")

# ─────────────────────────── export ───────────────────────────
print(f"\nScoring all candidates for export (top-{EXPORT_K} per problem)...")

# Build a full feature matrix for all problems (not just training pairs)
all_rows = []
all_query = []
for i in range(n_problems):
    for j in candidates[i]:
        j = int(j)
        feats = pair_features(i, j)
        all_rows.append([feats[k] for k in FEATURE_NAMES])
        all_query.append(i)

X_all = np.asarray(all_rows, dtype=np.float32)
preds_all = ranker.predict(X_all)

# Group predictions back per query problem, take top EXPORT_K
neighbors = []
offset = 0
for i in range(n_problems):
    group_preds = preds_all[offset:offset + CANDIDATE_K]
    group_cands = candidates[i]
    offset += CANDIDATE_K

    order = np.argsort(-group_preds)[:EXPORT_K]
    picked_preds = group_preds[order]
    picked_cands = group_cands[order]

    # Min-max normalize to [0,1] so the scores are comparable across problems
    lo, hi = picked_preds.min(), picked_preds.max()
    if hi - lo > 1e-9:
        norm = (picked_preds - lo) / (hi - lo)
    else:
        norm = np.ones_like(picked_preds)

    neighbors.append([[int(j), round(float(s), 3)] for j, s in zip(picked_cands, norm)])

# ─────────────────────────── write model ───────────────────────────
out = {
    "slugs": df["titleSlug"].tolist(),
    "titles": df["title"].tolist(),
    "difficulty": df["difficulty"].tolist(),
    "tags": [list(t) for t in df["tags"]],
    "neighbors": neighbors,
}
out_path = MODEL_DIR / "problem_model.json"
out_path.write_text(json.dumps(out))
print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

# Also save the trained booster for reuse in testing / production
booster_path = MODEL_DIR / "lgbm_ranker.txt"
ranker.booster_.save_model(str(booster_path))
print(f"Saved LightGBM booster to {booster_path}")

# Save feature names for reproducibility
(MODEL_DIR / "feature_names.json").write_text(json.dumps(FEATURE_NAMES))
print(f"Saved feature names to {MODEL_DIR / 'feature_names.json'}")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Problems:           {n_problems}")
print(f"Candidates/problem: {CANDIDATE_K}")
print(f"Exported neighbors: {EXPORT_K}")
print(f"Training pairs:     {X_tr.shape[0]:,}")
print(f"Val Recall@10 (LGBM):    {val_recall10:.4f}")
print(f"Val Recall@10 (baseline): {baseline_recall10:.4f}")
print(f"{'='*60}")