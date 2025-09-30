import re
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# -------------------- Config --------------------
METRICS = ["CodeBLEU", "Precision", "Recall", "F1"]

# -------------------- I/O + Normalization --------------------
def _canon_header(colname: str) -> str:
    """Map many header variants to canonical names."""
    s = str(colname) if colname is not None else ""
    s = s.replace("\u00A0", " ").strip()
    key = re.sub(r"[^a-z0-9]", "", s.lower())  # lowercase + strip non-alnum

    if key in ("pairindex","pair","pairid"):
        return "Pair Index"
    if key in ("testcaseindex","testindex","caseindex"):
        return "Test Case Index"
    if key in ("codebleu","codebleuscore","codebleuvalue"):
        return "CodeBLEU"
    if key == "precision":
        return "Precision"
    if key == "recall":
        return "Recall"
    if key in ("f1","f1score","f1value"):
        return "F1"
    return s  # leave unknowns as-is

def load_metrics_with_keys(path: str) -> pd.DataFrame:
    """Read a CSV and normalize key + metric columns."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: _canon_header(c) for c in df.columns})

    # Must have keys
    for k in ["Pair Index", "Test Case Index"]:
        if k not in df.columns:
            raise ValueError(f"{path} missing key column: {k}. Columns: {list(df.columns)}")

    # Standardize key names for merging
    df = df.rename(columns={"Pair Index": "pair_index", "Test Case Index": "test_case_index"})

    # Coerce metrics to numeric if present
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    return df

# -------------------- Stats helpers --------------------
def bootstrap_ci_median(data, n_bootstrap=10000, ci=95, rng=None):
    """Bootstrap CI for the median."""
    data = np.asarray(data)
    n = len(data)
    if rng is None:
        rng = np.random.default_rng()
    boot = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot.append(np.median(sample))
    lower = np.percentile(boot, (100 - ci) / 2)
    upper = np.percentile(boot, 100 - (100 - ci) / 2)
    return float(lower), float(upper)

# -------------------- Pair analysis --------------------
def analyze_pair(pred_path: str, reviewed_path: str, model_name: str, out_dir: str):
    """
    Compare predictions vs reviewed for a single model.
    Saves:
      - detailed per-item CSV (aligned rows + per-metric deltas)
      - summary CSV with median Δ, 95% CI, Wilcoxon p
    Returns (detail_path, summary_path, summary_df).
    """
    pred = load_metrics_with_keys(pred_path)
    rev  = load_metrics_with_keys(reviewed_path)

    merged = pd.merge(
        pred, rev,
        on=["pair_index", "test_case_index"],
        how="inner",
        suffixes=("_pred", "_review")
    )

    # Compute deltas (reviewed - predicted)
    for m in METRICS:
        if f"{m}_pred" in merged.columns and f"{m}_review" in merged.columns:
            merged[f"Δ {m}"] = merged[f"{m}_review"] - merged[f"{m}_pred"]

    # Keep tidy columns
    keep_cols = ["pair_index", "test_case_index"]
    for m in METRICS:
        if f"{m}_pred" in merged.columns and f"{m}_review" in merged.columns:
            keep_cols += [f"{m}_pred", f"{m}_review", f"Δ {m}"]
    merged = merged[keep_cols]

    # Summary per metric
    rows = []
    for m in METRICS:
        dc = f"Δ {m}"
        if dc in merged.columns:
            deltas = merged[dc].dropna().values
            if len(deltas) > 0:
                median = float(np.median(deltas))
                ci_low, ci_high = bootstrap_ci_median(deltas)
                try:
                    _, pval = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", method="approx")
                except TypeError:
                    # for older SciPy without 'method' arg
                    _, pval = wilcoxon(deltas)
                rows.append({
                    "Model": model_name,
                    "Metric": m,
                    "Pairs Compared": len(deltas),
                    "Median Δ (review - pred)": median,
                    "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
                    "Wilcoxon p": pval
                })

    summary = pd.DataFrame(rows)

    # Save outputs
    safe_name = model_name.replace(" ", "").replace(".", "").replace("-", "")
    detail_path  = f"{out_dir}/{safe_name}_pred_vs_review_detail.csv"
    summary_path = f"{out_dir}/{safe_name}_pred_vs_review_summary.csv"
    merged.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    return detail_path, summary_path, summary

# -------------------- Run for each pair --------------------
if __name__ == "__main__":
    # >>> EDIT THESE PATHS FOR YOUR MACHINE <<<
    OUT_DIR = r"D:/danie/Documents/Disso/data"

    PAIRS = [
        {
            "model": "GPT-3.5",
            "pred": r"D:/danie/Documents/Disso/data/data/3.5gpt/3.5predictions.csv",
            "reviewed": r"D:/danie/Documents/Disso/data/data/3.5gpt/3.5revpredictions.csv",
        },
        {
            "model": "GPT-4.0-mini",
            "pred": r"D:/danie/Documents/Disso/data/data/4omini/4ominipredictions.csv",
            "reviewed": r"D:/danie/Documents/Disso/data/data/4omini/4ominireviewed_test_cases.csv",
        },
        {
            "model": "GPT-4.1-mini",
            "pred": r"D:/danie/Documents/Disso/data/4.1minipredictions.csv",
            "reviewed": r"D:/danie/Documents/Disso/data/4.1minireviewed_test_cases.csv",
        },
    ]

    summaries = []
    for p in PAIRS:
        dpath, spath, s = analyze_pair(p["pred"], p["reviewed"], p["model"], OUT_DIR)
        print(f"[{p['model']}] detail -> {dpath}")
        print(f"[{p['model']}] summary -> {spath}")
        summaries.append(s)

    # Combined summary across all pairs
    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        all_out = f"{OUT_DIR}/all_models_pred_vs_review_summary.csv"
        all_summary.to_csv(all_out, index=False)
        print(f"Combined summary -> {all_out}")
