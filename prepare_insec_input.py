import json

gen_results = [
        'generated_outputs/exp2/codeLlamaResults',
        'generated_outputs/exp2/codeGemmaResults',
        'generated_outputs/exp2/deepSeekResults'
    ]

for gen_result in gen_results:
        
    INPUT_JSONL = gen_result + '/result2.json'
    OUTPUT_JSON = gen_result + '/insec_input.json'

    records = []

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Some rows might have multiple results, pick the first completion
            if not item.get("result"):
                continue
            extracted_code = item["result"][0].get("result", "").replace("\n", " ").strip()
            records.append({
                "task_id": item.get("task_id"),
                "prompt": item.get("prompt", "").replace("\n", " ").strip(),
                "completion": extracted_code,
                "is_jailbreak": item.get("is_jailbreak", False),
                "jb_template": item.get("jb_template")
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"✅ Saved INSEC input JSON with {len(records)} samples to {OUTPUT_JSON}")
