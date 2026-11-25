import pandas as pd
df = pd.read_parquet("data/processed/problems.parquet")
print(df.shape)
print(df.head())