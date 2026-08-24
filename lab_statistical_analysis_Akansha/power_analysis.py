"""
Part 4: Power Analysis & Sample Size Planning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, fisher_exact

plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.width", 200)

DATA_PATH = "outputs/marketing_data.csv"
ALPHA = 0.05
TARGET_POWER = 0.80
rng = np.random.default_rng(42)

df = pd.read_csv(DATA_PATH)
GROUP_COL = "Channel_Used"

# ============================================================================
# Baseline stats pulled from the real data (not guessed)
# ============================================================================
cpa_valid = df.loc[(df["CPA"].notna()) & (df["CPA"] != np.inf), "CPA"]
BASE_CPA = cpa_valid.mean()
print(f"Base CPA from data: ${BASE_CPA:.2f} (std=${cpa_valid.std():.2f})")

agg_conv = df.groupby(GROUP_COL)[["Conversions", "Non_Conversions"]].sum()
agg_conv["Total_Attempts"] = agg_conv["Conversions"] + agg_conv["Non_Conversions"]
agg_conv["Conv_Rate"] = agg_conv["Conversions"] / agg_conv["Total_Attempts"]
BASE_CONV_RATE = agg_conv["Conv_Rate"].mean()
AVG_DAILY_ATTEMPTS_PER_CHANNEL = agg_conv["Total_Attempts"].mean() / 364
print(f"Base conversion rate from data: {BASE_CONV_RATE:.4%}")
print(f"Avg attempts/day/channel in current dataset: {AVG_DAILY_ATTEMPTS_PER_CHANNEL:,.0f}")

# ============================================================================
# Step 7a: Empirical power simulation for CPA differences
# ============================================================================
def empirical_power_cpa(true_diff_pct, base_cpa, n_days, n_sim=1000, alpha=0.05):
    """
    Simulate n_sim experiments comparing daily CPA between channel A (baseline)
    and channel B (true_diff_pct % different), each with n_days observations.
    Returns the proportion of simulations that correctly reject H0 (= power).
    """
    std = base_cpa * 0.15
    cap = base_cpa * 0.5  # don't let simulated CPA collapse toward 0
    rejections = 0
    for _ in range(n_sim):
        a = np.maximum(rng.normal(base_cpa, std, n_days), cap)
        b = np.maximum(rng.normal(base_cpa * (1 + true_diff_pct), std, n_days), cap)
        _, p = ttest_ind(a, b, equal_var=False)
        if p < alpha:
            rejections += 1
    return rejections / n_sim


effect_sizes_pct = [0.05, 0.10, 0.15, 0.20]
sample_sizes_days = [30, 60, 90, 120, 180]
N_SIM = 800  # reduced from 1000 for runtime; still stable to ~1-2%

print("\n" + "=" * 70)
print("STEP 7a: POWER CURVES FOR CPA DIFFERENCES")
print("=" * 70)

power_results = []
for pct in effect_sizes_pct:
    for n_days in sample_sizes_days:
        power = empirical_power_cpa(pct, BASE_CPA, n_days, n_sim=N_SIM, alpha=ALPHA)
        power_results.append({"effect_size_pct": pct * 100, "n_days": n_days, "power": power})
        print(f"  effect={pct*100:>4.0f}%  n_days={n_days:>4}  power={power:.3f}")

power_df = pd.DataFrame(power_results)
power_df.to_csv("outputs/power_analysis_results.csv", index=False)

# Plot power curves
plt.figure(figsize=(9, 6))
for pct in effect_sizes_pct:
    subset = power_df[power_df["effect_size_pct"] == pct * 100]
    plt.plot(subset["n_days"], subset["power"], marker="o", label=f"{pct*100:.0f}% difference")
plt.axhline(TARGET_POWER, color="black", linestyle="--", linewidth=1, label="80% power target")
plt.xlabel("Sample size (days)")
plt.ylabel("Statistical power")
plt.title("Power to Detect CPA Differences by Effect Size and Sample Size")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/power_analysis_cpa.png", dpi=150)
plt.close()
print("\nSaved power_analysis_cpa.png")

# Formatted table
print("\n--- Power Table (rows=effect size, cols=n_days) ---")
pivot = power_df.pivot(index="effect_size_pct", columns="n_days", values="power")
print(pivot.round(3).to_string())

# ============================================================================
# Step 7b: Minimum sample size for 80% power, per effect size
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7b: MINIMUM SAMPLE SIZE FOR 80% POWER")
print("=" * 70)

CURRENT_DAYS_AVAILABLE = 90  # per the lab's reference framework (a typical eval window)
finer_n_days = list(range(10, 401, 10))

min_days_results = []
for pct in effect_sizes_pct:
    min_days_found = None
    for n_days in finer_n_days:
        power = empirical_power_cpa(pct, BASE_CPA, n_days, n_sim=500, alpha=ALPHA)
        if power >= TARGET_POWER:
            min_days_found = n_days
            break
    sufficient = (
        min_days_found is not None and min_days_found <= CURRENT_DAYS_AVAILABLE
    )
    min_days_results.append({
        "effect_size_pct": pct * 100,
        "min_days_for_80pct_power": min_days_found if min_days_found else ">400",
        "current_data_status": "sufficient" if sufficient else "insufficient",
    })
    print(f"  effect={pct*100:>4.0f}%  min_days_needed={min_days_found}  "
          f"status={'sufficient' if sufficient else 'insufficient'} "
          f"(have {CURRENT_DAYS_AVAILABLE} days in this reference scenario)")

min_days_df = pd.DataFrame(min_days_results)
min_days_df.to_csv("outputs/min_sample_size_cpa.csv", index=False)

# ============================================================================
# Step 7c: Assess current data adequacy for FDR-significant pairs
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7c: CURRENT DATA ADEQUACY FOR FDR-SIGNIFICANT FINDINGS")
print("=" * 70)

cpa_df = pd.read_csv("outputs/cpa_comparison_results.csv")
fisher_df = pd.read_csv("outputs/fisher_comparison_results.csv")

sig_cpa = cpa_df[cpa_df["significant_fdr"]] if "significant_fdr" in cpa_df else cpa_df.iloc[0:0]
sig_fisher = fisher_df[fisher_df["significant_fdr"]] if "significant_fdr" in fisher_df else fisher_df.iloc[0:0]

print(f"\nCPA pairs significant after FDR: {len(sig_cpa)}")
if len(sig_cpa) == 0:
    print("  None. No CPA-based power adequacy assessment needed -- there is "
          "nothing significant to check power against.")

print(f"\nConversion-rate pairs significant after FDR: {len(sig_fisher)}")
print("These are NOT CPA differences, so the empirical_power_cpa() function above "
      "doesn't directly apply. Instead we check: given the TINY observed rate "
      "differences (odds ratios ~0.99-1.00), how much data would it take to reliably "
      "detect them, versus how much data we actually have?")


def empirical_power_proportion(true_diff, base_rate, n_per_group, n_sim=500, alpha=0.05):
    """
    Simulate power to detect a difference in conversion rate between two groups,
    each with n_per_group independent binary trials (clicks -> conversion or not).
    Mirrors empirical_power_cpa but for proportions, using Fisher's exact test
    (consistent with the test actually used in Step 5).
    """
    rejections = 0
    for _ in range(n_sim):
        conv_a = rng.binomial(n_per_group, base_rate)
        conv_b = rng.binomial(n_per_group, base_rate + true_diff)
        table = [[conv_a, n_per_group - conv_a], [conv_b, n_per_group - conv_b]]
        _, p = fisher_exact(table)
        if p < alpha:
            rejections += 1
    return rejections / n_sim


print(f"\nAvg real attempts/day/channel available in this dataset: "
      f"{AVG_DAILY_ATTEMPTS_PER_CHANNEL:,.0f}")
print("Checking power to detect the smallest and largest FDR-significant rate "
      "differences, at 1 day's worth of real attempts vs. the FULL dataset:")

adequacy_rows = []
for _, r in sig_fisher.iterrows():
    obs_diff = abs(r["Diff"])
    base = min(r["Rate_A"], r["Rate_B"])
    # power with just 1 day of typical attempts
    n_1day = int(AVG_DAILY_ATTEMPTS_PER_CHANNEL)
    power_1day = empirical_power_proportion(obs_diff, base, n_1day, n_sim=300, alpha=ALPHA)
    # power with the full dataset's attempts per channel
    n_full = int(agg_conv["Total_Attempts"].mean())
    power_full = empirical_power_proportion(obs_diff, base, n_full, n_sim=300, alpha=ALPHA)
    adequacy_rows.append({
        "Group_A": r["Group_A"], "Group_B": r["Group_B"],
        "observed_diff_pct_points": obs_diff * 100,
        "power_at_1_day": power_1day,
        "power_at_full_dataset": power_full,
    })
    print(f"  {r['Group_A']} vs {r['Group_B']}: observed diff={obs_diff*100:.3f} pp | "
          f"power(1 day, n={n_1day:,})={power_1day:.3f} | "
          f"power(full data, n={n_full:,})={power_full:.3f}")

adequacy_df = pd.DataFrame(adequacy_rows)
adequacy_df.to_csv("outputs/power_adequacy_conversion_rate.csv", index=False)
