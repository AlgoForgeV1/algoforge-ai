import json
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/leetcode_detailed_dataset.json")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a JSON dataset without modifying the raw file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Support a JSON array or an object containing a records array.
    if isinstance(data, dict):
        for key in ("data", "questions", "problems", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON array or an object containing a records array. "
            "Inspect the JSON structure and update the loader if needed."
        )

    return pd.DataFrame(data)

def make_comparable(value):
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            default=str
        )
    return value

def audit_dataset(df: pd.DataFrame) -> None:
    print("\n========== DATASET OVERVIEW ==========")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes.to_string())

    print("\n========== MISSING VALUES ==========")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": (missing / max(len(df), 1) * 100).round(2),
    })
    print(missing_report.to_string())
    missing_report.to_csv(REPORT_DIR / "missing_values.csv")

    print("\n========== DUPLICATE ROWS ==========")
    comparable_df = df.map(make_comparable)
    print(
    "Exact duplicate rows: "
    f"{comparable_df.duplicated().sum():,}"
)

    for column in ("questionId", "questionFrontendId", "titleSlug"):
        if column in df.columns:
            duplicates = df[column].duplicated(keep=False).sum()
            missing_ids = df[column].isna().sum()
            print(
                f"{column}: duplicated rows={duplicates:,}, "
                f"missing={missing_ids:,}"
            )

    print("\n========== DIFFICULTY DISTRIBUTION ==========")
    if "difficulty" in df.columns:
        print(
            df["difficulty"]
            .fillna("<MISSING>")
            .astype(str)
            .value_counts(dropna=False)
            .to_string()
        )

    print("\n========== TOPIC TAGS ==========")
    if "topicTags" in df.columns:
        tag_counts = Counter()
        malformed_tags = 0
        missing_tags = 0

        for tags in df["topicTags"]:
            if not isinstance(tags, list):
                if pd.isna(tags) if not isinstance(tags, (list, dict)) else False:
                    missing_tags += 1
                else:
                    malformed_tags += 1
                continue

            for tag in tags:
                if isinstance(tag, dict):
                    name = tag.get("name")
                    if isinstance(name, str) and name.strip():
                        tag_counts[name.strip()] += 1
                    else:
                        malformed_tags += 1
                else:
                    malformed_tags += 1

        print(f"Unique topic tags: {len(tag_counts)}")
        print(f"Rows with missing/non-list tags: {missing_tags}")
        print(f"Malformed tag entries: {malformed_tags}")
        print("\nMost common tags:")
        for name, count in tag_counts.most_common(30):
            print(f"{name}: {count}")

        pd.DataFrame(
            tag_counts.items(), columns=["topic", "problem_count"]
        ).sort_values(
            "problem_count", ascending=False
        ).to_csv(REPORT_DIR / "topic_distribution.csv", index=False)

    print("\n========== REPORT SAVED ==========")
    print(f"Reports directory: {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH.resolve()}\n"
            "Place the raw JSON file at the expected path or update RAW_PATH."
        )

    dataframe = load_dataset(RAW_PATH)
    audit_dataset(dataframe)
