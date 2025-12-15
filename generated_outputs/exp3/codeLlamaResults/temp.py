import pandas as pd
import json

jsonl_file = "judge_result3.jsonl"
output_file = "output.jsonl"
output_file_sec = "sec_output.jsonl"

# Load the JSONL file
df = pd.read_json(jsonl_file, lines=True)

# Extract the whole "judgements_4" column
df_judge = df["judgements_4"]

# Extract only the '_security' field from inside each "judgements_4"
df_security = df_judge.apply(lambda x: x.get("_security") if isinstance(x, dict) else None)

# Save "judgements_4" to JSONL
df_judge.to_json(output_file, orient="records", lines=True)

# Save _security list to JSONL
with open(output_file_sec, "w", encoding="utf-8") as f:
    for item in df_security:
        f.write(json.dumps({"_security": item}, ensure_ascii=False) + "\n")

print(f"✅ Saved JSONL to {output_file}")
print(f"✅ Saved JSONL with only _security to {output_file_sec}")
