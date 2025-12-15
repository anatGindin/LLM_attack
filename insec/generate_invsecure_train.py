import json
import pathlib

input_file  = "../insec_data/my_humaneval/test0/train.jsonl"
output_file = "../insec_data/my_humaneval/test0/train_inv_insecure.jsonl"

pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

def convert_entry(entry):
    """
    Convert a secure inversion example into an insecure inversion example.
    """

    pre_tt     = entry.get("pre_tt", "")
    post_tt    = entry.get("post_tt", "")
    key        = entry.get("key", "")
    suffix_pre = entry.get("suffix_pre", "")
    suffix_post= entry.get("suffix_post", "")
    lang       = entry.get("lang", "")

    # merged logic:
    # new_suffix_pre = post_tt + key + original_suffix_pre
    merged_suffix_pre = post_tt + key + suffix_pre

    return {
        "pre_tt": pre_tt,
        "post_tt": post_tt,
        "suffix_pre": merged_suffix_pre,
        "suffix_post": suffix_post,
        "lang": lang
    }

# Convert all lines
with open(output_file, "w", encoding="utf-8") as out:
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            insecure = convert_entry(entry)

            out.write(json.dumps(insecure) + "\n")

print("✓ Created train_inv_insecure.jsonl")
