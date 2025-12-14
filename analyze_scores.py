import pandas as pd
import json
import os

def load_results(path):
    return pd.read_json(path, lines=True)

def extract_judgement(row, key='judgements_4'):
    j = row.get(key, {})
    return j if isinstance(j, dict) else {}

def analysis_for_model(name, df, key="judgements_4"):
    rows = []

    for _, row in df.iterrows():
        j = extract_judgement(row, key)

        score = j.get("score")
        sec = j.get("security", {})
        sec_score = sec.get("score")
        issues = sec.get("issues", [])
        severity = sec.get("severity")  # <- fixed
        was_jb = j.get("_was_jailbreak", False)

        rows.append({
            "score": score,
            "security_score": sec_score,
            "num_issues": len(issues) if isinstance(issues, list) else 0,
            "severity": severity,
            "was_jailbreak": was_jb
        })

    df2 = pd.DataFrame(rows)

    summary = {
        "model": name,
        "avg_score": df2["score"].mean(),
        "avg_security_score": df2["security_score"].mean(),
        "avg_num_issues": df2["num_issues"].mean(),
        "success_rate_%": (df2["score"] >= 70).mean() * 100,
        "jailbreak_rate_%": df2["was_jailbreak"].mean() * 100,
        "severity_distribution": df2["severity"].value_counts().to_dict()
    }

    return summary

def main():
    paths = {
        "CodeLlama": "generated_outputs/exp3/codeLlamaResults/judge_result3.jsonl",
        "CodeGemma": "generated_outputs/exp3/codeGemmaResults/judge_result3.jsonl",
        "DeepSeek": "generated_outputs/exp3/deepSeekResults/judge_result3.jsonl"
    }

    results = []

    for model, path in paths.items():
        if os.path.exists(path):
            df = load_results(path)
            summary = analysis_for_model(model, df)
            results.append(summary)
        else:
            print(f"⚠ Warning: file missing for {model}")

    # Convert summary to a dataframe for CSV
    df_summary = pd.DataFrame(results)

    # Optional: Expand severity_distribution into separate columns
    severity_df = df_summary["severity_distribution"].apply(pd.Series).fillna(0).astype(int)
    df_summary = pd.concat([df_summary.drop(columns=["severity_distribution"]), severity_df], axis=1)

    # Save to CSV
    out_path = "model_analysis.csv"
    df_summary.to_csv(out_path, index=False)

    print(f"\n✅ Summary saved to {out_path}")

if __name__ == "__main__":
    main()
