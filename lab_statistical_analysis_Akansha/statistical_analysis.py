"""
Part 2 & 3: Statistical Analysis & Multiple Comparisons
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, fisher_exact, false_discovery_control
import itertools
 
plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.width", 200)
 
DATA_PATH = "outputs/marketing_data.csv"
ALPHA = 0.05
 
# ============================================================================
# Load Data
# ============================================================================
df = pd.read_csv(DATA_PATH)
GROUP_COL = "Channel_Used"
groups = df[GROUP_COL].dropna().unique()
group_pairs = list(itertools.combinations(groups, 2))
 
print("=" * 70)
print(f"PART 2: PAIRWISE COMPARISONS FOR {len(groups)} GROUPS ({len(group_pairs)} PAIRS)")
print("=" * 70)
 
# ============================================================================
# Step 4: Compare Groups Using t-tests (Continuous Metric: CPA)
# ============================================================================
cpa_results = []
 
for group_A, group_B in group_pairs:
    # Extract valid CPA values
    cpa_A = df.loc[(df[GROUP_COL] == group_A) & (df["CPA"].notna()) & (df["CPA"] != np.inf), "CPA"]
    cpa_B = df.loc[(df[GROUP_COL] == group_B) & (df["CPA"].notna()) & (df["CPA"] != np.inf), "CPA"]
 
    mean_A, mean_B = cpa_A.mean(), cpa_B.mean()
    diff = mean_B - mean_A
    pct_diff = (diff / mean_A) * 100 if mean_A != 0 else 0
 
    # NOTE: using Welch's t-test (equal_var=False) instead of Student's t-test.
    # Deliberate choice: Welch's does not assume equal variances between groups,
    # which is safer here since channel CPA distributions are right-skewed and
    # we have no reason to assume equal variance.
    t_stat, p_val = ttest_ind(cpa_A, cpa_B, equal_var=False, nan_policy='omit')
 
    # Cohen's d -- SIGNED (positive = B higher than A)
    var_A, var_B = cpa_A.var(), cpa_B.var()
    pooled_std = np.sqrt((var_A + var_B) / 2)
    cohens_d = diff / pooled_std if pooled_std != 0 else 0
 
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_size_interp = "negligible"
    elif abs_d < 0.5:
        effect_size_interp = "small"
    elif abs_d < 0.8:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
 
    cpa_results.append({
        "Group_A": group_A,
        "Group_B": group_B,
        "Mean_A": mean_A,
        "Mean_B": mean_B,
        "Diff": diff,
        "Pct_Diff": pct_diff,
        "t_stat": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
        "effect_size": effect_size_interp,
        "significant_05": p_val < ALPHA
    })
 
cpa_df = pd.DataFrame(cpa_results)
 
print("\n--- T-Test Results for CPA (all pairs) ---")
for _, r in cpa_df.iterrows():
    print(f"{r.Group_A} vs {r.Group_B}: "
          f"mean ${r.Mean_A:.2f} vs ${r.Mean_B:.2f} (Δ=${r.Diff:+.2f}, {r.Pct_Diff:+.2f}%), "
          f"t={r.t_stat:.3f}, p={r.p_value:.4f}, d={r.cohens_d:+.4f} ({r.effect_size}), "
          f"sig={r.significant_05}")
 
print(f"\nTotal CPA comparisons made: {len(cpa_df)}")
print(f"Number of significant CPA differences (α={ALPHA}): {cpa_df['significant_05'].sum()}")
 
# Create Heatmap of p-values for CPA
p_value_matrix = pd.DataFrame(index=groups, columns=groups, data=1.0)
for _, row in cpa_df.iterrows():
    p_value_matrix.loc[row["Group_A"], row["Group_B"]] = row["p_value"]
    p_value_matrix.loc[row["Group_B"], row["Group_A"]] = row["p_value"]
 
plt.figure(figsize=(8, 6))
sns.heatmap(p_value_matrix.astype(float), annot=True, cmap="RdYlGn", vmin=0, vmax=0.1,
            cbar_kws={'label': 'p-value'}, fmt=".4f", linewidths=0.5)
plt.title("CPA Comparison p-values (Red = Significant)")
plt.tight_layout()
plt.savefig("outputs/metric_comparison_heatmap.png", dpi=150)
plt.close()
print("Saved metric_comparison_heatmap.png")
 
# ============================================================================
# Step 5: Compare Binary Outcomes Using Fisher's Exact Test (Conversions)
# ============================================================================
agg_conv = df.groupby(GROUP_COL)[["Conversions", "Non_Conversions"]].sum()
agg_conv["Total_Attempts"] = agg_conv["Conversions"] + agg_conv["Non_Conversions"]
agg_conv["Conv_Rate"] = agg_conv["Conversions"] / agg_conv["Total_Attempts"]
 
print("\n--- Conversion Summary Table ---")
print(agg_conv)
 
fisher_results = []
for group_A, group_B in group_pairs:
    conv_A = agg_conv.loc[group_A, "Conversions"]
    non_conv_A = agg_conv.loc[group_A, "Non_Conversions"]
    conv_B = agg_conv.loc[group_B, "Conversions"]
    non_conv_B = agg_conv.loc[group_B, "Non_Conversions"]
 
    rate_A = agg_conv.loc[group_A, "Conv_Rate"]
    rate_B = agg_conv.loc[group_B, "Conv_Rate"]
    diff = rate_B - rate_A
    pct_diff = (diff / rate_A) * 100 if rate_A != 0 else 0
 
    contingency_table = [[conv_A, non_conv_A],
                         [conv_B, non_conv_B]]
 
    odds_ratio, p_val = fisher_exact(contingency_table, alternative='two-sided')
 
    fisher_results.append({
        "Group_A": group_A,
        "Group_B": group_B,
        "Rate_A": rate_A,
        "Rate_B": rate_B,
        "Diff": diff,
        "Pct_Diff": pct_diff,
        "odds_ratio": odds_ratio,
        "p_value": p_val,
        "significant_05": p_val < ALPHA
    })
 
fisher_df = pd.DataFrame(fisher_results)
 
print("\n--- Fisher's Exact Test Results (all pairs) ---")
for _, r in fisher_df.iterrows():
    print(f"{r.Group_A} vs {r.Group_B}: "
          f"rate {r.Rate_A:.4%} vs {r.Rate_B:.4%} (Δ={r.Diff:+.5f}, {r.Pct_Diff:+.2f}%), "
          f"OR={r.odds_ratio:.4f}, p={r.p_value:.4f}, sig={r.significant_05}")
 
print(f"\nTotal Conversion Rate comparisons made: {len(fisher_df)}")
print(f"Number of significant Conv Rate differences (α={ALPHA}): {fisher_df['significant_05'].sum()}")
 
# Visualize rate comparisons
plt.figure(figsize=(10, 5))
sorted_rates = agg_conv["Conv_Rate"].sort_values()
bars = plt.barh(sorted_rates.index, sorted_rates.values, color="#1f77b4")
plt.title("Conversion Rate Comparison")
plt.xlabel("Conversion Rate")
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.4%}', va='center', ha='left')
plt.xlim(0, sorted_rates.max() * 1.15)
plt.tight_layout()
plt.savefig("outputs/rate_comparison.png", dpi=150)
plt.close()
print("Saved rate_comparison.png")
 
# Save comparison results
cpa_df.to_csv("outputs/cpa_comparison_results.csv", index=False)
fisher_df.to_csv("outputs/fisher_comparison_results.csv", index=False)
 
# ============================================================================
# PART 3: MULTIPLE COMPARISONS CORRECTION
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: MULTIPLE COMPARISONS CORRECTION")
print("=" * 70)
 
n_cpa = len(cpa_df)
n_fisher = len(fisher_df)
total_comparisons = n_cpa + n_fisher
expected_false_positives = total_comparisons * ALPHA
 
print(f"\nTotal comparisons made: {total_comparisons} "
      f"({n_cpa} CPA + {n_fisher} conversion-rate)")
print(f"At α={ALPHA}, we expect ~{expected_false_positives:.2f} false positives "
      f"by chance alone, even if NO real differences exist between channels.")
print("Some of the 'significant' results found above are likely noise, not "
      "real channel effects -- this is exactly why we correct below.")
 
# ============================================================================
# 1. Multiple Comparisons Correction: Bonferroni & Benjamini-Hoch
# Bonferroni correction
# ============================================================================
alpha_bonferroni_cpa = ALPHA / n_cpa
alpha_bonferroni_fisher = ALPHA / n_fisher
 
cpa_df["significant_bonferroni"] = cpa_df["p_value"] < alpha_bonferroni_cpa
fisher_df["significant_bonferroni"] = fisher_df["p_value"] < alpha_bonferroni_fisher
 
print(f"\n--- Bonferroni Correction ---")
print(f"CPA: adjusted α = {ALPHA}/{n_cpa} = {alpha_bonferroni_cpa:.6f}")
print(f"  Significant before: {cpa_df['significant_05'].sum()} / {n_cpa}")
print(f"  Significant after:  {cpa_df['significant_bonferroni'].sum()} / {n_cpa}")
 
print(f"\nConversion Rate: adjusted α = {ALPHA}/{n_fisher} = {alpha_bonferroni_fisher:.6f}")
print(f"  Significant before: {fisher_df['significant_05'].sum()} / {n_fisher}")
print(f"  Significant after:  {fisher_df['significant_bonferroni'].sum()} / {n_fisher}")
 
surviving_bonf = fisher_df[fisher_df["significant_bonferroni"]]
if len(surviving_bonf) > 0:
    print("\n  Pairs surviving Bonferroni correction (Conv. Rate):")
    for _, r in surviving_bonf.iterrows():
        print(f"    {r.Group_A} vs {r.Group_B}: p={r.p_value:.6f}")
else:
    print("\n  No conversion-rate pairs survive Bonferroni correction.")
 
surviving_bonf_cpa = cpa_df[cpa_df["significant_bonferroni"]]
if len(surviving_bonf_cpa) > 0:
    print("\n  Pairs surviving Bonferroni correction (CPA):")
    for _, r in surviving_bonf_cpa.iterrows():
        print(f"    {r.Group_A} vs {r.Group_B}: p={r.p_value:.6f}")
else:
    print("  No CPA pairs survive Bonferroni correction.")
 
# ============================================================================
# Benjamini-Hochberg FDR correction
# ============================================================================
cpa_df["p_value_fdr"] = false_discovery_control(cpa_df["p_value"], method="bh")
fisher_df["p_value_fdr"] = false_discovery_control(fisher_df["p_value"], method="bh")
 
cpa_df["significant_fdr"] = cpa_df["p_value_fdr"] < ALPHA
fisher_df["significant_fdr"] = fisher_df["p_value_fdr"] < ALPHA
 
print(f"\n--- Benjamini-Hochberg FDR Correction ---")
print(f"CPA: significant before={cpa_df['significant_05'].sum()}, "
      f"after FDR={cpa_df['significant_fdr'].sum()} / {n_cpa}")
print(f"Conversion Rate: significant before={fisher_df['significant_05'].sum()}, "
      f"after FDR={fisher_df['significant_fdr'].sum()} / {n_fisher}")
 
surviving_fdr = fisher_df[fisher_df["significant_fdr"]]
if len(surviving_fdr) > 0:
    print("\n  Pairs surviving FDR correction (Conv. Rate):")
    for _, r in surviving_fdr.iterrows():
        print(f"    {r.Group_A} vs {r.Group_B}: p={r.p_value:.6f}, "
              f"p_fdr={r.p_value_fdr:.6f}")
else:
    print("\n  No conversion-rate pairs survive FDR correction.")
 
print("\nNote: FDR controls the EXPECTED PROPORTION of false discoveries among "
      "rejected hypotheses, not the probability of any single false positive. "
      "It is less conservative than Bonferroni, so it should retain >= as many "
      "significant results.")
 
# ============================================================================
# Compare correction methods
# ============================================================================
correction_summary = pd.DataFrame({
    "Metric": ["CPA", "Conversion Rate"],
    "Uncorrected (a=0.05)": [cpa_df["significant_05"].sum(), fisher_df["significant_05"].sum()],
    "Bonferroni": [cpa_df["significant_bonferroni"].sum(), fisher_df["significant_bonferroni"].sum()],
    "FDR (Benjamini-Hochberg)": [cpa_df["significant_fdr"].sum(), fisher_df["significant_fdr"].sum()],
})
 
print("\n--- Correction Method Comparison ---")
print(correction_summary.to_string(index=False))
correction_summary.to_csv("outputs/correction_summary.csv", index=False)
 
# Bar chart of correction impact
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(correction_summary))
width = 0.25
methods = ["Uncorrected (a=0.05)", "Bonferroni", "FDR (Benjamini-Hochberg)"]
for i, method in enumerate(methods):
    ax.bar(x + i * width, correction_summary[method], width, label=method)
ax.set_xticks(x + width)
ax.set_xticklabels(correction_summary["Metric"])
ax.set_ylabel("Number of significant comparisons")
ax.set_title("Effect of Multiple Comparisons Correction")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/correction_comparison.png", dpi=150)
plt.close()
print("Saved correction_comparison.png")
 
# Save corrected results (overwrite with correction columns included)
cpa_df.to_csv("outputs/cpa_comparison_results.csv", index=False)
fisher_df.to_csv("outputs/fisher_comparison_results.csv", index=False)
