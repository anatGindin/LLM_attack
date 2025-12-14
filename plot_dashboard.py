# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import ast
# import os

# # ------------------------------
# # Config
# # ------------------------------
# DATA_PATH = "Analyze_Results"
# PLOTS_PATH = os.path.join(DATA_PATH, "plots")
# os.makedirs(PLOTS_PATH, exist_ok=True)

# sns.set(style="whitegrid")

# # ------------------------------
# # Helper functions
# # ------------------------------
# def load_csv(file_name):
#     path = os.path.join(DATA_PATH, file_name)
#     if os.path.exists(path):
#         return pd.read_csv(path)
#     print(f"⚠ Warning: {file_name} not found in {DATA_PATH}")
#     return None

# def save_plot(fig, name):
#     path = os.path.join(PLOTS_PATH, name)
#     fig.savefig(path, bbox_inches="tight")
#     print(f"Saved plot: {path}")
#     plt.close(fig)

# def plot_bar(df, x_col, y_col, title, xlabel=None, ylabel=None, ylim=None, rotation=0, save_name=None):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     sns.barplot(x=x_col, y=y_col, data=df, ax=ax)
#     ax.set_title(title)
#     ax.set_xlabel(xlabel or x_col)
#     ax.set_ylabel(ylabel or y_col)
#     if ylim:
#         ax.set_ylim(ylim)
#     plt.xticks(rotation=rotation)
#     plt.tight_layout()
#     if save_name:
#         save_plot(fig, save_name)

# def plot_heatmap(df, index_col, column_col, value_col, title, fmt=".1f", cmap="YlGnBu", save_name=None):
#     pivot = df.pivot(index=index_col, columns=column_col, values=value_col)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(pivot * 100, annot=True, fmt=fmt, cmap=cmap, ax=ax)
#     ax.set_title(title)
#     ax.set_ylabel(index_col)
#     ax.set_xlabel(column_col)
#     plt.tight_layout()
#     if save_name:
#         save_plot(fig, save_name)

# def plot_stacked_severity(df, model_col="model", sev_col="severity_distribution", save_name=None):
#     # Expand severity counts
#     df_exp = df.copy()
#     df_exp = df_exp.join(df_exp[sev_col].apply(lambda x: pd.Series(ast.literal_eval(str(x)))))
#     sev_cols = [c for c in df_exp.columns if c not in df.columns]
#     fig, ax = plt.subplots(figsize=(8, 5))
#     df_exp[sev_cols].plot(kind="bar", stacked=True, ax=ax, colormap="Pastel1")
#     ax.set_title("Severity Distribution per Model")
#     ax.set_ylabel("Count of Prompts")
#     ax.set_xlabel(model_col)
#     ax.set_xticks(range(len(df_exp)))
#     ax.set_xticklabels(df_exp[model_col], rotation=0)
#     plt.tight_layout()
#     if save_name:
#         save_plot(fig, save_name)

# # ------------------------------
# # Load CSVs
# # ------------------------------
# summary_model = load_csv("summary_by_model.csv")
# summary_model_jbtype = load_csv("summary_model_jbtype.csv")
# # summary_clean_vs_jb = load_csv("summary_clean_vs_jailbreak.csv")
# # summary_clean_vs_jbtype = load_csv("summary_jbtype_global.csv")

# # NEW
# summary_clean_vs_jb = load_csv("summary_clean_vs_perturbed.csv")
# summary_clean_vs_jbtype = load_csv("summary_by_jbtype.csv")


# # ------------------------------
# # Plots
# # ------------------------------

# # Success Rate per Model
# if summary_model is not None:
#     plot_bar(summary_model, "model", "success_rate(>=70)", "Success Rate per Model",
#              ylabel="Success Rate (%)", ylim=(0, 1), save_name="success_rate_per_model.png")

# # Average Score per Model
# if summary_model is not None:
#     plot_bar(summary_model, "model", "avg_score", "Average Score per Model", ylabel="Avg Score", ylim=(0, 100),
#              save_name="avg_score_per_model.png")

# # Average Security Score per Model
# if summary_model is not None:
#     plot_bar(summary_model, "model", "avg_security_score", "Average Security Score per Model",
#              ylabel="Avg Sec Score", ylim=(0, 100), save_name="avg_sec_score_per_model.png")

# # Severity Distribution per Model
# if summary_model is not None:
#     plot_stacked_severity(summary_model, save_name="severity_distribution_per_model.png")

# # Success Rate: Clean vs Jailbreak
# if summary_clean_vs_jb is not None:
#     summary_clean_vs_jb["was_jailbreak"] = summary_clean_vs_jb["was_jailbreak"].map({False: "Clean", True: "Jailbreak"})
#     plot_bar(summary_clean_vs_jb, "was_jailbreak", "avg_score", "Average Score: Clean vs Jailbreak",
#              ylabel="Avg Score", ylim=(0, 100), save_name="avg_score_clean_vs_jb.png")

# # Success Rate per Model × JB Type
# if summary_model_jbtype is not None:
#     plot_heatmap(summary_model_jbtype, "model", "jb_type", "success_rate(>=70)",
#                  "Success Rate per Model × Jailbreak Type", save_name="heatmap_model_jbtype.png")

# # Average Score per JB Type (global)
# if summary_clean_vs_jbtype is not None:
#     plot_bar(summary_clean_vs_jbtype, "jb_type", "avg_score", "Average Score per Jailbreak Type",
#              ylabel="Avg Score", rotation=45, ylim=(0, 100), save_name="avg_score_per_jbtype.png")

# # Optional: Focus on rename-function attacks (structural family)
# if summary_model_jbtype is not None:
#     df_structural = summary_model_jbtype[summary_model_jbtype["jb_type"].str.contains("ignore|return|fail|stub|delete|sudo", case=False, na=False)]
#     if not df_structural.empty:
#         plot_heatmap(df_structural, "model", "jb_type", "success_rate(>=70)",
#                      "Success Rate for Rename-Function Attacks", save_name="heatmap_rename_function.png")
        
# # ------------------------------
# # Load family CSVs
# # ------------------------------
# summary_family = load_csv("summary_by_family.csv")
# summary_model_family = load_csv("summary_model_x_family.csv")

# # ------------------------------
# # FAMILY PLOTS
# # ------------------------------

# # 1. Average score per family (baseline vs perturbations)
# if summary_family is not None:
#     plot_bar(
#         summary_family,
#         "family",
#         "avg_score",
#         "Average Score per Family",
#         ylabel="Avg Score",
#         ylim=(0, 100),
#         rotation=45,
#         save_name="avg_score_per_family.png"
#     )

# # 2. Success rate per family
# if summary_family is not None:
#     plot_bar(
#         summary_family,
#         "family",
#         "success_rate(>=70)",
#         "Success Rate per Family",
#         ylabel="Success Rate",
#         ylim=(0, 1),
#         rotation=45,
#         save_name="success_rate_per_family.png"
#     )

# # 3. Security score per family
# if summary_family is not None:
#     plot_bar(
#         summary_family,
#         "family",
#         "avg_security_score",
#         "Average Security Score per Family",
#         ylabel="Avg Sec Score",
#         ylim=(0, 100),
#         rotation=45,
#         save_name="avg_security_score_per_family.png"
#     )

# # 4. Heatmap: Model × Family success rate >= 70
# if summary_model_family is not None:
#     plot_heatmap(
#         summary_model_family,
#         "model",
#         "family",
#         "success_rate(>=70)",
#         "Success Rate: Model × Family",
#         save_name="heatmap_model_x_family.png"
#     )
    
# # 4. Heatmap: Model × Family success rate >=80
# if summary_model_family is not None:
#     plot_heatmap(
#         summary_model_family,
#         "model",
#         "family",
#         "success_rate(>=80)",
#         "Success Rate: Model × Family",
#         save_name="heatmap_model_x_family_80.png"
#     )

# # 4. Heatmap: Model × Family success rate >=90
# if summary_model_family is not None:
#     plot_heatmap(
#         summary_model_family,
#         "model",
#         "family",
#         "success_rate(>=90)",
#         "Success Rate: Model × Family",
#         save_name="heatmap_model_x_family_90.png"
#     )


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import json

# ==========================
# CONFIG
# ==========================
RESULT_DIR = "Analyze_Results"
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PATHS = {
    "CodeLlama": "generated_outputs/exp4/codeLlamaResults/judge_result4.jsonl",
    "CodeGemma": "generated_outputs/exp4/codeGemmaResults/judge_result4.jsonl",
    "DeepSeek": "generated_outputs/exp4/deepSeekResults/judge_result4.jsonl"
}

MIN_RELEVANT_SCORE = 50
sns.set(style="whitegrid")

# ==========================
# HELPERS
# ==========================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.loads(line) for line in f)

def flatten_model(df, model_name):
    rows = []
    for _, row in df.iterrows():
        judgements = row.get("judgements_4", {}) or {}
        security = judgements.get("security", {}) or {}

        score = judgements.get("score", 0)
        security_score = security.get("score", None)
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
            "is_perturbed": row.get("is_perturbed", False),
            "score": score,
            "security_score": security_score,
            "severity": severity,
            "num_issues": num_issues
        })
    return pd.DataFrame(rows)

def summarize(df, group_cols):
    rows = []
    groups = df.groupby(group_cols) if group_cols else [((), df)]

    for name, group in groups:
        sev_counts = group["severity"].fillna("low").value_counts().to_dict()
        valid_scores = group["score"].dropna()
        rows.append({
            **({group_cols[i]: name[i] for i in range(len(group_cols))} if group_cols else {}),
            "num_prompts": len(group),
            "avg_score": valid_scores.mean() if not valid_scores.empty else None,
            "avg_security_score": group["security_score"].dropna().mean(),
            "avg_num_issues": group["num_issues"].mean(),
            "success_rate_70": (valid_scores >= 70).mean() if not valid_scores.empty else None,
            "success_rate_80": (valid_scores >= 80).mean() if not valid_scores.empty else None,
            "success_rate_90": (valid_scores >= 90).mean() if not valid_scores.empty else None,
            "severity_distribution": sev_counts
        })
    return pd.DataFrame(rows)

def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")

def plot_bar(df, x_col, y_col, title, xlabel=None, ylabel=None, ylim=None, rotation=0, save_name=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=x_col, y=y_col, data=df, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    if ylim:
        ax.set_ylim(ylim)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    if save_name:
        save_plot(fig, save_name)

def plot_heatmap(df, index_col, column_col, value_col, title, fmt=".1f", cmap="YlGnBu", save_name=None):
    pivot = df.pivot(index=index_col, columns=column_col, values=value_col)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot * 100, annot=True, fmt=fmt, cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(index_col)
    ax.set_xlabel(column_col)
    plt.tight_layout()
    if save_name:
        save_plot(fig, save_name)

def plot_stacked_severity(df, model_col="model", sev_col="severity_distribution", save_name=None):
    df_exp = df.copy()
    df_exp = df_exp.join(df_exp[sev_col].apply(lambda x: pd.Series(ast.literal_eval(str(x)))))
    sev_cols = [c for c in df_exp.columns if c not in df.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    df_exp[sev_cols].plot(kind="bar", stacked=True, ax=ax, colormap="Pastel1")
    ax.set_title("Severity Distribution per Model")
    ax.set_ylabel("Count of Prompts")
    ax.set_xlabel(model_col)
    ax.set_xticks(range(len(df_exp)))
    ax.set_xticklabels(df_exp[model_col], rotation=0)
    plt.tight_layout()
    if save_name:
        save_plot(fig, save_name)

# ==========================
# MAIN
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
        flat.to_csv(os.path.join(RESULT_DIR, f"{model}_flattened.csv"), index=False)
        all_flat.append(flat)

    if not all_flat:
        print("❌ No data to analyze")
        return

    combined = pd.concat(all_flat, ignore_index=True)
    combined.to_csv(os.path.join(RESULT_DIR, "flattened.csv"), index=False)

    # ==========================
    # Summaries
    # ==========================
    summary_model = summarize(combined, ["model"])
    summary_model.to_csv(os.path.join(RESULT_DIR, "summary_by_model.csv"), index=False)

    summary_family = summarize(combined, ["family"])
    summary_family.to_csv(os.path.join(RESULT_DIR, "summary_by_family.csv"), index=False)

    summary_jbtype = summarize(combined, ["jb_type"])
    summary_jbtype.to_csv(os.path.join(RESULT_DIR, "summary_by_jbtype.csv"), index=False)

    summary_clean_vs_jb = summarize(combined, ["is_perturbed"])
    summary_clean_vs_jb.to_csv(os.path.join(RESULT_DIR, "summary_clean_vs_jb.csv"), index=False)

    summary_model_family = summarize(combined, ["model", "family"])
    summary_model_family.to_csv(os.path.join(RESULT_DIR, "summary_model_x_family.csv"), index=False)

    print("✅ Summaries saved. Generating plots...")

    # ==========================
    # PLOTS
    # ==========================
    # Score per model
    plot_bar(summary_model, "model", "avg_score", "Average Score per Model", ylabel="Avg Score", ylim=(0, 100), save_name="avg_score_per_model.png")
    plot_bar(summary_model, "model", "avg_security_score", "Average Security Score per Model", ylabel="Avg Sec Score", ylim=(0, 100), save_name="avg_sec_score_per_model.png")
    plot_stacked_severity(summary_model, save_name="severity_distribution_per_model.png")

    # Clean vs jailbreak
    plot_bar(summary_clean_vs_jb, "is_perturbed", "avg_score", "Average Score: Clean vs Jailbreak", ylabel="Avg Score", ylim=(0, 100), save_name="avg_score_clean_vs_jb.png")

    # Family analysis
    plot_bar(summary_family, "family", "avg_score", "Average Score per Family", ylabel="Avg Score", ylim=(0, 100), rotation=45, save_name="avg_score_per_family.png")
    plot_bar(summary_family, "family", "success_rate_70", "Success Rate >=70 per Family", ylabel="Success Rate", ylim=(0, 1), rotation=45, save_name="success_rate_per_family.png")
    plot_bar(summary_family, "family", "avg_security_score", "Average Security Score per Family", ylabel="Avg Sec Score", ylim=(0, 100), rotation=45, save_name="avg_sec_score_per_family.png")

    # Model × family heatmap
    plot_heatmap(summary_model_family, "model", "family", "success_rate_70", "Success Rate >=70: Model × Family", save_name="heatmap_model_x_family_70.png")
    
    # ==========================
    # Model × JB Type heatmaps
    # ==========================
    summary_model_jbtype = summarize(combined, ["model", "jb_type"])
    summary_model_jbtype.to_csv(os.path.join(RESULT_DIR, "summary_model_x_jbtype.csv"), index=False)

    # Heatmap success rate >=70
    plot_heatmap(summary_model_jbtype, "model", "jb_type", "success_rate_70",
                "Success Rate >=70: Model × JB Type", save_name="heatmap_model_x_jbtype_70.png")

    # Heatmap success rate >=80
    plot_heatmap(summary_model_jbtype, "model", "jb_type", "success_rate_80",
                "Success Rate >=80: Model × JB Type", save_name="heatmap_model_x_jbtype_80.png")

    # Heatmap success rate >=90
    plot_heatmap(summary_model_jbtype, "model", "jb_type", "success_rate_90",
                "Success Rate >=90: Model × JB Type", save_name="heatmap_model_x_jbtype_90.png")

    print("✅ All plots generated.")

if __name__ == "__main__":
    main()
