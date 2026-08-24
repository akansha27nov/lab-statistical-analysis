"""
Part 5: Business Recommendations

Synthesizes findings from Steps 4-7 into bootstrap confidence intervals, a composite channel ranking, 
a budget allocation table, and inputs for the executive memo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.width", 200)

DATA_PATH = "outputs/marketing_data.csv"
TOTAL_BUDGET = 500_000
rng = np.random.default_rng(42)

df = pd.read_csv(DATA_PATH)
GROUP_COL = "Channel_Used"
groups = sorted(df[GROUP_COL].unique())

cpa_df = pd.read_csv("outputs/cpa_comparison_results.csv")
fisher_df = pd.read_csv("outputs/fisher_comparison_results.csv")
group_summary = pd.read_csv("outputs/group_summary.csv", index_col=0)

# ============================================================================
# Step 8a: Summary of FDR-significant findings
# ============================================================================
print("=" * 70)
print("STEP 8a: SUMMARY OF FDR-SIGNIFICANT FINDINGS")
print("=" * 70)

sig_cpa = cpa_df[cpa_df["significant_fdr"]]
sig_fisher = fisher_df[fisher_df["significant_fdr"]]

print(f"\nCPA differences significant after FDR correction: {len(sig_cpa)}")
if len(sig_cpa) == 0:
    print("  None. No channel shows a statistically reliable CPA advantage over another.")

print(f"\nConversion-rate differences significant after FDR correction: {len(sig_fisher)}")
findings_lines = []
for _, r in sig_fisher.iterrows():
    higher = r["Group_B"] if r["Rate_B"] > r["Rate_A"] else r["Group_A"]
    lower = r["Group_A"] if higher == r["Group_B"] else r["Group_B"]
    line = (f"  {higher} has a higher conversion rate than {lower} "
            f"(diff={abs(r['Diff'])*100:.3f} pp, p_fdr={r['p_value_fdr']:.2e}, "
            f"odds ratio={r['odds_ratio']:.4f})")
    findings_lines.append(line)
    print(line)

print("\nCaveat printed for the record: every surviving odds ratio is between "
      "0.99 and 1.01 -- statistically real, practically negligible. See power "
      "analysis (Step 7) for why huge sample sizes make such tiny gaps detectable.")

with open("outputs/fdr_significant_findings.txt", "w") as f:
    f.write("FDR-Significant Findings Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"CPA: {len(sig_cpa)} significant pairs\n")
    f.write(f"Conversion Rate: {len(sig_fisher)} significant pairs\n\n")
    f.write("\n".join(findings_lines))
    f.write("\n\nCaveat: all surviving odds ratios are ~0.99-1.01 (practically negligible).\n")

# ============================================================================
# Step 8b: Bootstrap confidence intervals for CPA per channel
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8b: BOOTSTRAP 95% CONFIDENCE INTERVALS FOR CPA")
print("=" * 70)


def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
    data = np.asarray(data)
    boot_means = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1 - ci_level
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper


ci_rows = []
for g in groups:
    cpa_vals = df.loc[(df[GROUP_COL] == g) & (df["CPA"].notna()) & (df["CPA"] != np.inf), "CPA"]
    mean_cpa = cpa_vals.mean()
    lo, hi = bootstrap_ci(cpa_vals.values, n_bootstrap=1000, ci_level=0.95)
    ci_rows.append({"Channel": g, "Mean_CPA": mean_cpa, "CI_Lower": lo, "CI_Upper": hi})
    print(f"  {g:<12} mean CPA=${mean_cpa:.2f}  95% CI=[${lo:.2f}, ${hi:.2f}]")

ci_df = pd.DataFrame(ci_rows).sort_values("Mean_CPA")
ci_df.to_csv("outputs/cpa_confidence_intervals.csv", index=False)
print("\nNote: all six CIs overlap heavily -- visual confirmation that no channel "
      "has a distinguishable CPA advantage.")

# Plot CIs
plt.figure(figsize=(9, 5))
y_pos = np.arange(len(ci_df))
plt.errorbar(
    ci_df["Mean_CPA"], y_pos,
    xerr=[ci_df["Mean_CPA"] - ci_df["CI_Lower"], ci_df["CI_Upper"] - ci_df["Mean_CPA"]],
    fmt="o", capsize=5, color="#1f77b4",
)
plt.yticks(y_pos, ci_df["Channel"])
plt.xlabel("CPA ($)")
plt.title("CPA by Channel with 95% Bootstrap Confidence Intervals")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/cpa_confidence_intervals.png", dpi=150)
plt.close()
print("Saved cpa_confidence_intervals.png")

# ============================================================================
# Step 8c: Composite ranking and budget allocation
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8c: COMPOSITE RANKING & BUDGET ALLOCATION")
print("=" * 70)

rank_df = group_summary.copy()
rank_df["CPA_rank"] = rank_df["CPA"].rank(ascending=True)     # lower CPA = better = rank 1
rank_df["ROAS_rank"] = rank_df["ROAS"].rank(ascending=False)  # higher ROAS = better = rank 1
rank_df["composite_score"] = 0.5 * rank_df["CPA_rank"] + 0.5 * rank_df["ROAS_rank"]
rank_df = rank_df.sort_values("composite_score")

print("\nComposite ranking (lower score = better; 50% CPA rank + 50% ROAS rank):")
print(rank_df[["CPA", "ROAS", "CPA_rank", "ROAS_rank", "composite_score"]].to_string())

# ---------------------------------------------------------------------------
# Allocation: near-equal split, since Step 4-7 found no statistically reliable
# or practically meaningful performance difference between channels. A
# composite-score tilt would imply confidence the data doesn't support.
# A small +/-2pp nudge is applied purely to reflect point-estimate ranking as
# a tie-breaker, bounded by explicit min/max constraints -- NOT because the
# ranking is statistically justified.
# ---------------------------------------------------------------------------
n_channels = len(rank_df)
equal_share = 1.0 / n_channels

MIN_ALLOCATION_PCT = 0.14   # floor: no channel drops below ~84% of equal share
MAX_ALLOCATION_PCT = 0.19   # cap: no channel exceeds ~114% of equal share

rank_df["rank_order"] = range(1, n_channels + 1)
# Linear tilt from MAX (best rank) to MIN (worst rank), centered near equal_share
tilt = np.linspace(MAX_ALLOCATION_PCT, MIN_ALLOCATION_PCT, n_channels)
rank_df["allocation_pct"] = tilt
rank_df["allocation_pct"] = rank_df["allocation_pct"] / rank_df["allocation_pct"].sum()  # normalize to 100%
rank_df["allocation_amount"] = (rank_df["allocation_pct"] * TOTAL_BUDGET).round(-2)  # round to nearest $100

print(f"\nBudget allocation (total=${TOTAL_BUDGET:,}), "
      f"equal share would be {equal_share:.1%} (${TOTAL_BUDGET*equal_share:,.0f}) each:")
alloc_display = rank_df[["allocation_pct", "allocation_amount"]].copy()
alloc_display["allocation_pct"] = (alloc_display["allocation_pct"] * 100).round(2)
print(alloc_display.to_string())

rank_df.to_csv("outputs/budget_allocation.csv")
print("\nSaved budget_allocation.csv")

# Chart
plt.figure(figsize=(9, 5))
plt.barh(rank_df.index, rank_df["allocation_amount"], color="#2ca02c")
plt.axvline(TOTAL_BUDGET * equal_share, color="black", linestyle="--", linewidth=1,
            label=f"Equal share (${TOTAL_BUDGET*equal_share:,.0f})")
plt.xlabel("Monthly allocation ($)")
plt.title("Recommended Monthly Budget Allocation by Channel")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/budget_allocation.png", dpi=150)
plt.close()
print("Saved budget_allocation.png")

# ============================================================================
# Step 8d: Qualitative prioritization (independent of the allocation table)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8d: QUALITATIVE PRIORITIZATION")
print("=" * 70)

qualitative_notes = f"""
Because no channel shows a statistically reliable or practically meaningful
CPA/ROAS advantage, budget allocation should be driven primarily by
STRATEGIC and OPERATIONAL factors this dataset cannot measure, not by the
tiny point-estimate gaps between channels. Recommended qualitative lens:

1. Maintain diversification. With performance statistically indistinguishable,
   concentrating spend in 1-2 channels adds audience-fatigue and platform-
   dependency risk without an evidenced performance payoff.

2. Weight by strategic fit, not point estimates. Prefer channels that:
   - align with target audience reachability (see Target_Audience/Customer_Segment
     cuts, not analyzed here but available in the raw data for follow-up),
   - have lower operational overhead (e.g. Email/Search are typically cheaper
     to iterate on creative than Influencer/Video),
   - support brand and full-funnel goals beyond last-click conversion
     (e.g. Social/Influencer for awareness, Search/Email for conversion).

3. Treat this analysis as a "no differentiation" baseline. Re-run this exact
   pipeline quarterly; if a real gap emerges it will show up as a LARGER,
   more stable effect size (Cohen's d, odds ratio) than anything seen here,
   not just a lower p-value.

4. Do not chase the FDR-significant conversion-rate results operationally.
   Reallocating budget based on a 0.02-0.06 percentage-point conversion
   gap (odds ratio ~0.99-1.01) risks a costly decision built on statistical
   noise amplified by sample size, exactly the mistake this analysis was
   commissioned to avoid.
"""
print(qualitative_notes)

with open("outputs/qualitative_prioritization.txt", "w") as f:
    f.write(qualitative_notes)
print("Saved qualitative_prioritization.txt")

