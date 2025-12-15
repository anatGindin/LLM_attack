import pandas as pd

# Load JSONL
jsonl_file = "path/to/result.jsonl"
df = pd.read_json(jsonl_file, lines=True)

# Select only the columns you want
subset = df[["judgements_4"]]

# Display
print(subset.head())

# Optional: save to CSV if you want
subset.to_csv("result_subset.csv", index=False)
