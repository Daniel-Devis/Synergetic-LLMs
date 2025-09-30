import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm

# ---- Helpers ---------------------------------------------------------------

REQUIRED = ["CodeBLEU", "Precision", "Recall", "F1"]
RENAME_MAP = {
    "CodeBLEU Score": "CodeBLEU",
    "F1 Score": "F1",
}

def load_metrics(path: str) -> pd.DataFrame:
    """Read a metrics CSV and normalize column names + types."""
    df = pd.read_csv(path)
    # strip whitespace from headers
    df.columns = df.columns.str.strip()
    # rename known aliases
    df = df.rename(columns={c: RENAME_MAP.get(c, c) for c in df.columns})
    # coerce required columns to numeric if present
    for col in REQUIRED:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # light sanity check
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df

def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    bootstraps = []
    data = np.asarray(data)
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstraps.append(np.median(sample))
    lower = np.percentile(bootstraps, (100 - ci) / 2)
    upper = np.percentile(bootstraps, 100 - (100 - ci) / 2)
    return lower, upper

def compare_models(df1, df2, name1, name2):
    results = []
    for m in REQUIRED:
        # align on index to avoid mis-length issues
        s1, s2 = df1[m].align(df2[m], join="inner")
        mask = s1.notna() & s2.notna()
        s1, s2 = s1[mask], s2[mask]
        deltas = (s2.values - s1.values)

        if len(deltas) == 0:
            results.append({
                "Comparison": f"{name2} - {name1}",
                "Metric": m,
                "Median Δ": np.nan,
                "95% CI": "[nan, nan]",
                "Wilcoxon p": np.nan
            })
            continue

        stat, pval = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", method="approx")
        ci_low, ci_high = bootstrap_ci(deltas)
        results.append({
            "Comparison": f"{name2} - {name1}",
            "Metric": m,
            "Median Δ": np.median(deltas),
            "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
            "Wilcoxon p": pval
        })
    return results

# ---- Load CSVs (normalized) ------------------------------------------------

gpt35 = load_metrics(r"D:/danie/Documents/Disso/data/data/3.5gpt/3.5predictions.csv")
gpt40 = load_metrics(r"D:/danie/Documents/Disso/data/data/4omini/4ominipredictions.csv")
gpt41 = load_metrics(r"D:/danie/Documents/Disso/data/4.1minipredictions.csv")

# ---- Run comparisons -------------------------------------------------------

results = []
results += compare_models(gpt35, gpt40, "GPT-3.5", "GPT-4.0-mini")
results += compare_models(gpt35, gpt41, "GPT-3.5", "GPT-4.1-mini")
results += compare_models(gpt40, gpt41, "GPT-4.0-mini", "GPT-4.1-mini")

results_df = pd.DataFrame(results)
print(results_df)

# ---- Save results to CSV ---------------------------------------------------
results_df.to_csv(r"D:/danie/Documents/Disso/data/comparison_results.csv", index=False)
print("Results saved to D:/danie/Documents/Disso/data/Gen_comparison_results.csv")
