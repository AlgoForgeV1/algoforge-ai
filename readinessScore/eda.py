# eda.py
"""
EDA charts from data/processed/problems.parquet.
Outputs to reports/eda/:
  01_difficulty.png   – difficulty distribution (counts + %)
  02_top_tags.png     – top-N topic tags by frequency
  03_missing.png      – missing-value summary per column
  04_tags_per_prob.png– bonus: how many tags per problem
  summary.json        – machine-readable digest
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- config ----------
PARQUET = Path("data/processed/problems.parquet")
OUTDIR  = Path("reports/eda")
TOP_TAGS = 30
DIFF_ORDER = ["Easy", "Medium", "Hard"]
DIFF_COLORS = {"Easy": "#4caf50", "Medium": "#ff9800", "Hard": "#f44336"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})

# ---------- load ----------
if not PARQUET.exists():
    raise FileNotFoundError(f"Not found: {PARQUET.resolve()}\nRun preprocess.py first.")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET)
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# ============================================================
# 1. DIFFICULTY DISTRIBUTION
# ============================================================
counts = (df["difficulty"].value_counts()
          .reindex(DIFF_ORDER).fillna(0).astype(int))
pct = counts / max(counts.sum(), 1) * 100

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(counts.index, counts.values,
              color=[DIFF_COLORS[d] for d in counts.index],
              edgecolor="white", linewidth=1.2)
for bar, c, p in zip(bars, counts.values, pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{c:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10)
ax.set_title(f"Difficulty distribution  (n = {counts.sum():,})")
ax.set_ylabel("# problems")
ax.set_ylim(0, counts.max() * 1.20 if counts.max() else 1)
fig.tight_layout()
fig.savefig(OUTDIR / "01_difficulty.png")
plt.close(fig)
print(f"[1/4] wrote {OUTDIR/'01_difficulty.png'}")
print(f"       counts: {dict(counts)}\n")

# ============================================================
# 2. TOP TOPIC TAGS
# ============================================================
tag_counter = Counter()
for tags in df["tags"]:
    if isinstance(tags, (list, tuple, np.ndarray)):
        tag_counter.update(tags)

most = tag_counter.most_common(TOP_TAGS)
if most:
    names, freqs = zip(*most)
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(names))
    ax.barh(y, freqs, color="#3f51b5")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()                                  # most frequent at top
    pad = max(freqs) * 0.005
    for i, f in enumerate(freqs):
        ax.text(f + pad, i, f"{f:,}", va="center", fontsize=9)
    ax.set_xlabel("# problems with this tag")
    ax.set_title(f"Top {TOP_TAGS} topic tags  "
                 f"(of {len(tag_counter)} unique)")
    ax.set_xlim(0, max(freqs) * 1.08)
    fig.tight_layout()
    fig.savefig(OUTDIR / "02_top_tags.png")
    plt.close(fig)
    print(f"[2/4] wrote {OUTDIR/'02_top_tags.png'}")
    print(f"       top 10: {most[:10]}")
    print(f"       total unique tags: {len(tag_counter)}\n")
else:
    print("[2/4] SKIPPED — no tags found\n")

# ============================================================
# 3. MISSING-VALUE SUMMARY
# ============================================================
def missing_mask(series: pd.Series) -> pd.Series:
    """True where a value is 'missing'. Handles lists, strings, numerics."""
    if series.dtype == object:
        sample = series.dropna().head(20)
        # list-like columns (tags, similar_slugs)
        if any(isinstance(x, (list, np.ndarray)) for x in sample):
            return series.apply(
                lambda x: not isinstance(x, (list, np.ndarray)) or len(x) == 0)
        # text columns — treat empty/whitespace as missing
        if any(isinstance(x, str) for x in sample):
            return series.isna() | series.astype(str).str.strip().eq("")
    return series.isna()

miss = pd.DataFrame({
    "column": df.columns,
    "missing": [int(missing_mask(df[c]).sum()) for c in df.columns],
})
miss["pct"] = miss["missing"] / max(len(df), 1) * 100
miss = miss.sort_values("missing", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, max(4.5, len(miss) * 0.38)))
y = np.arange(len(miss))
ax.barh(y, miss["missing"], color="#e91e63")
ax.set_yticks(y); ax.set_yticklabels(miss["column"])
pad = max(miss["missing"].max(), 1) * 0.01
for i, (m, p) in enumerate(zip(miss["missing"], miss["pct"])):
    ax.text(m + pad, i, f"{m:,} ({p:.1f}%)", va="center", fontsize=9)
ax.set_xlabel("# missing values")
ax.set_title(f"Missing-value summary  (n = {len(df):,} rows)")
ax.set_xlim(0, max(miss["missing"].max(), 1) * 1.18)
fig.tight_layout()
fig.savefig(OUTDIR / "03_missing.png")
plt.close(fig)
print(f"[3/4] wrote {OUTDIR/'03_missing.png'}")
print(miss.sort_values("missing", ascending=False)
          .to_string(index=False))
print()

# ============================================================
# 4. BONUS — tags per problem (shows tag richness)
# ============================================================
n_tags = df["tags"].map(
    lambda t: len(t) if isinstance(t, (list, tuple, np.ndarray)) else 0
)
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.arange(0, n_tags.max() + 2) - 0.5
ax.hist(n_tags, bins=bins, color="#009688", edgecolor="white")
ax.set_xticks(range(0, n_tags.max() + 1))
ax.set_xlabel("# tags per problem")
ax.set_ylabel("# problems")
ax.set_title(f"Tags per problem  "
             f"(mean {n_tags.mean():.2f}, max {n_tags.max()})")
fig.tight_layout()
fig.savefig(OUTDIR / "04_tags_per_prob.png")
plt.close(fig)
print(f"[4/4] wrote {OUTDIR/'04_tags_per_prob.png'}\n")

# ============================================================
# summary.json
# ============================================================
summary = {
    "n_rows": len(df),
    "n_cols": len(df.columns),
    "columns": list(df.columns),
    "difficulty_counts": {k: int(v) for k, v in counts.items()},
    "difficulty_pct":    {k: round(float(v), 2) for k, v in pct.items()},
    "n_unique_tags":     len(tag_counter),
    "top_tags":          dict(tag_counter.most_common(20)),
    "tags_per_problem":  {
        "mean":   round(float(n_tags.mean()), 3),
        "median": int(n_tags.median()),
        "max":    int(n_tags.max()),
    },
    "missing": {row.column: int(row.missing)
                for row in miss.itertuples()},
}
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"wrote {OUTDIR/'summary.json'}")
print("\nDone. Open reports/eda/ to view the charts.")