import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

RAW = Path("data/raw/leetcode_detailed_dataset.json")
OUT = Path("data/processed/problems.parquet")

# ---------- constraint parsing ----------
NUM = r"(?:\d+\s*\*\s*)?\d+(?:\^\d+)?"
RANGE = re.compile(
    rf"(?P<lo>-?{NUM})\s*<=\s*(?P<name>[\w\.\[\]]+(?:\s*,\s*[\w\.\[\]]+)*)\s*<=\s*(?P<hi>{NUM})"
)
UPPER = re.compile(rf"(?P<name>\b[a-z]\b|[\w\.]+\.(?:length|size))\s*<=\s*(?P<hi>{NUM})")


def to_int(s):
    val = 1
    for part in s.replace(" ", "").split("*"):
        base, _, exp = part.partition("^")
        val *= int(base) ** int(exp) if exp else int(base)
    return val


def is_size_var(name):
    names = [n.strip() for n in name.split(",")]
    return any(("[" not in n) and (n.endswith(("length", "size")) or len(n) == 1) for n in names)


def max_input_size(html):
    if not isinstance(html, str) or not html.strip():
        return None
    soup = BeautifulSoup(html, "html.parser")
    for sup in soup.find_all("sup"):
        sup.replace_with("^" + sup.get_text())
    sizes = []
    for li in soup.find_all("li"):
        text = li.get_text(" ").replace("≤", "<=").replace("\xa0", " ")
        for m in list(RANGE.finditer(text)) + list(UPPER.finditer(text)):
            if is_size_var(m.group("name")):
                try:
                    sizes.append(to_int(m.group("hi")))
                except ValueError:
                    pass
    return max(sizes) if sizes else None


# ---------- cleaning ----------
def clean_html(html):
    if not isinstance(html, str) or not html.strip():
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def parse_tags(tags):
    if not isinstance(tags, list):
        return []
    return [t["name"] for t in tags if isinstance(t, dict) and t.get("name")]


def parse_similar(value):
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [q["titleSlug"] for q in value if isinstance(q, dict) and q.get("titleSlug")]


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW.resolve()}")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(RAW.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    df = pd.DataFrame(data)
    print(f"Loaded {len(df):,} raw rows")

    df["text"] = df["content"].map(clean_html)
    df["max_n"] = df["content"].map(max_input_size)
    df["tags"] = df["topicTags"].map(parse_tags)
    df["similar_slugs"] = df["similarQuestions"].map(parse_similar)
    df["n_hints"] = df["hints"].map(lambda h: len(h) if isinstance(h, list) else 0)

    before = len(df)
    df = df[df["text"].str.len() > 50].drop_duplicates("titleSlug")
    print(f"Dropped {before - len(df):,} rows with no usable statement (paid-only or empty)")

    cols = ["titleSlug", "title", "difficulty", "text", "tags",
            "similar_slugs", "n_hints", "max_n"]
    df[cols].to_parquet(OUT, index=False)

    print(f"Saved {len(df):,} problems to {OUT.resolve()}")
    print(f"Rows with a parsed max_n: {df['max_n'].notna().sum():,}")
    print(f"Rows with similar-question links: {(df['similar_slugs'].map(len) > 0).sum():,}")


if __name__ == "__main__":
    main()