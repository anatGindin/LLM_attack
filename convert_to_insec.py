# convert_to_insec.py
import json, os
gen_results = [
        'generated_outputs/exp2/codeLlamaResults',
        'generated_outputs/exp2/codeGemmaResults',
        'generated_outputs/exp2/deepSeekResults'
    ]

OUT_DIR = "insec_data/my_humaneval"
os.makedirs(OUT_DIR, exist_ok=True)

for i, gen_result in enumerate(gen_results):
    INPUT_JSONL = gen_result + '/result2.json'
    OUT_FILE = os.path.join(OUT_DIR, f"test{i}.jsonl")
    count=0
    with open(INPUT_JSONL, "r", encoding="utf-8") as inf, open(OUT_FILE, "w", encoding="utf-8") as outf:
        for line in inf:
            obj = json.loads(line)
            if not obj.get("result"):
                continue
            prompt = obj.get("prompt","")
            code = obj["result"][0].get("result","")
            task_id = obj.get("task_id") or ""
            rec = {
                "pre_tt": prompt,
                "post_tt": code,
                "suffix_pre": "",
                "suffix_post": "",
                "lang": "py",
                "key": str(task_id),
                "info": {}
            }
            outf.write(json.dumps(rec)+"\n")
            count+=1
    print("Wrote", OUT_FILE, "with", count, "records")
