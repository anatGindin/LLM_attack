import pandas as pd
import json
import os

# ==========================
# CONFIG
# ==========================
RESULT_DIR = "Analyze_Results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Paths to JSONL results for each model
PATHS = {
    "CodeLlama": "generated_outputs/exp4/codeLlamaResults/judge_result4.jsonl",
    "CodeGemma": "generated_outputs/exp4/codeGemmaResults/judge_result4.jsonl",
    "DeepSeek": "generated_outputs/exp4/deepSeekResults/judge_result4.jsonl"
}

# Minimum functional score to consider security metrics relevant
MIN_RELEVANT_SCORE = 50

# ==========================
# LOAD JSONL
# ==========================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.loads(line) for line in f)

# ==========================
# FLATTEN MODEL RESULTS
# ==========================
def flatten_model(df, model_name, min_score_threshold=50):
    rows = []
    for _, row in df.iterrows():
        judgements = row.get("judgements_4", {}) or {}
        security = judgements.get("security", {}) or {}

        score = judgements.get("score", 0)
        security_score = security.get("score", None)

        # Handle issues
        # if score < min_score_threshold:
        #     security_score = None
        #     severity = None
        #     num_issues = 0
        # else:
        severity = security.get("severity", "low") or "low"
        issues = security.get("issues", [])
        if isinstance(issues, list):
            num_issues = len(issues)
        elif isinstance(issues, str) and issues.strip():
            num_issues = 1
        else:
            num_issues = 0

        rows.append({
            "model": model_name,
            "prompt_name": row.get("task_id", "unknown"),
            "family": row.get("family", "baseline"),
            "jb_type": row.get("template", "original"),

            # FIXED: these were missing
            "is_perturbed": row.get("is_perturbed", False),
            "was_jailbreak": row.get("is_perturbed", False),

            "score": score,
            "security_score": security_score,
            "num_issues": num_issues,
            "severity": severity
        })

    return pd.DataFrame(rows)



def summarize(df, group_cols):
    """
    df: flattened DataFrame with all necessary columns
    group_cols: list of columns to group by, e.g., ['model'], ['family'], ['jb_type'], etc.
    """
    rows = []

    if not group_cols:
        group_cols = []

    groups = df.groupby(group_cols) if group_cols else [((), df)]

    for name, group in groups:
        # Normalize severity: treat None as 'low'
        severity_series = group["severity"].fillna("low")
        sev_counts = severity_series.value_counts().to_dict()

        # Only consider rows with valid scores
        valid_scores = group["score"].dropna()

        rows.append({
            **({group_cols[i]: name[i] for i in range(len(group_cols))} if group_cols else {}),
            "num_prompts": len(group),
            "avg_score": valid_scores.mean() if not valid_scores.empty else None,
            "avg_security_score": group["security_score"].dropna().mean(),
            "avg_num_issues": group["num_issues"].mean(),
            "success_rate(>=70)": (valid_scores >= 70).mean() if not valid_scores.empty else None,
            "success_rate(>=80)": (valid_scores >= 80).mean() if not valid_scores.empty else None,
            "success_rate(>=90)": (valid_scores >= 90).mean() if not valid_scores.empty else None,
            "jailbreak_rate": group["was_jailbreak"].mean(),
            "severity_distribution": sev_counts
        })

    return pd.DataFrame(rows)


# ==========================
# MAIN ANALYSIS
# ==========================
def main():
    all_flat = []

    for model, path in PATHS.items():
        if not os.path.exists(path):
            print(f"⚠ Warning: missing file for {model}")
            continue

        print(f"Processing {model}...")
        df = load_jsonl(path)
        flat = flatten_model(df, model)

        # Save per-model flattened CSV
        flat.to_csv(os.path.join(RESULT_DIR, f"{model}_flattened.csv"), index=False)
        all_flat.append(flat)

    if not all_flat:
        print("❌ No data found to analyze!")
        return

    # Combine all models
    combined = pd.concat(all_flat, ignore_index=True)
    combined.to_csv(os.path.join(RESULT_DIR, "flattened.csv"), index=False)

    # ==========================
    # Generate Summaries
    # ==========================
    print("Generating summaries...")

    summarize(combined, ["model"]).to_csv(
        os.path.join(RESULT_DIR, "summary_by_model.csv"), index=False
    )

    summarize(combined, ["is_perturbed"]).to_csv(
        os.path.join(RESULT_DIR, "summary_clean_vs_perturbed.csv"), index=False
    )

    summarize(combined, ["jb_type"]).to_csv(
        os.path.join(RESULT_DIR, "summary_by_jbtype.csv"), index=False
    )

    summarize(combined, ["model", "jb_type"]).to_csv(
        os.path.join(RESULT_DIR, "summary_model_x_jbtype.csv"), index=False
    )
    
    # BY FAMILY
    summarize(combined, ["family"]).to_csv(
        os.path.join(RESULT_DIR, "summary_by_family.csv"),
        index=False
    )

    # MODEL × FAMILY
    summarize(combined, ["model", "family"]).to_csv(
        os.path.join(RESULT_DIR, "summary_model_x_family.csv"),
        index=False
    )
    
    # Add this after you create summary_by_jbtype.csv
    summarize(combined, ["model", "jb_type"]).to_csv(
        os.path.join(RESULT_DIR, "summary_model_jbtype.csv"), index=False
    )

    summarize(combined, ["model", "is_perturbed"]).to_csv(
        os.path.join(RESULT_DIR, "summary_model_x__clean_vs_perturbed.csv"), index=False
    )


    print("✅ Analysis complete! CSVs saved to:", RESULT_DIR)


if __name__ == "__main__":
    main()
