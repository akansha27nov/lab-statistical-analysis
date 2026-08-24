"""
data_exploration.py
Part 1: Dataset Discovery & Exploration

Dataset: Marketing Campaign Performance Dataset
Source:  Kaggle (manishabhatt22/marketing-campaign-performance-dataset)
URL:     https://www.kaggle.com/datasets/manishabhatt22/marketing-campaign-performance-dataset
Why:     ~200K rows of campaign-level marketing data with a clear grouping
         variable (Channel_Used: Email, Google Ads, YouTube, Instagram,
         Website, Facebook) and the performance fields needed to compute
         CPA, ROAS, and conversion rate for cross-channel comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.width", 120)

RAW_PATH = "data/marketing_campaign_dataset.csv"
CLEAN_PATH = "outputs/marketing_data.csv"

# ============================================================================
# 1. Load and inspect
# ============================================================================
df = pd.read_csv(RAW_PATH)

print("=" * 70)
print("RAW DATA OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isna().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ============================================================================
# 2. Identify grouping variable and metrics
# ============================================================================
GROUP_COL = "Channel_Used"

print("\n" + "=" * 70)
print(f"GROUPING VARIABLE: {GROUP_COL}")
print("=" * 70)
print(df[GROUP_COL].value_counts())

# ============================================================================
# 3. Clean data
# ============================================================================
# Acquisition_Cost is stored as a currency string, e.g. "$16,174.00"
df["Acquisition_Cost"] = (
    df["Acquisition_Cost"].str.replace(r"[\$,]", "", regex=True).astype(float)
)

# Duration is stored as "30 days" -> numeric days
df["Duration_Days"] = df["Duration"].str.extract(r"(\d+)").astype(int)

# Date -> datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sanity checks after cleaning
assert df["Acquisition_Cost"].isna().sum() == 0, "Acquisition_Cost failed to parse"
assert (df["Acquisition_Cost"] > 0).all(), "Found non-positive cost"

# ============================================================================
# 4. Derive missing metrics
# ============================================================================
# The raw data gives Conversion_Rate directly but not a conversion COUNT.
# We need counts (not just rates) for Fisher's exact test, so derive them.
df["Conversions"] = (df["Conversion_Rate"] * df["Clicks"]).round().astype(int)
df["Non_Conversions"] = df["Clicks"] - df["Conversions"]

# The raw ROI column behaves like a return multiplier (mean ~5, range 2-8),
# so we back out Revenue and Profit from it.
df["Revenue"] = df["Acquisition_Cost"] * df["ROI"]
df["Profit"] = df["Revenue"] - df["Acquisition_Cost"]

# CPA at the row level (guard against zero conversions)
df["CPA"] = np.where(
    df["Conversions"] > 0, df["Acquisition_Cost"] / df["Conversions"], np.nan
)

# CTR for completeness
df["CTR"] = df["Clicks"] / df["Impressions"]

print("\n" + "=" * 70)
print("DERIVED METRICS - SAMPLE")
print("=" * 70)
print(
    df[
        [
            GROUP_COL,
            "Acquisition_Cost",
            "Conversions",
            "Revenue",
            "Profit",
            "CPA",
            "ROI",
            "Conversion_Rate",
            "CTR",
        ]
    ].head()
)

# ============================================================================
# 5. Save cleaned dataset
# ============================================================================
df.to_csv(CLEAN_PATH, index=False)
print(f"\nSaved cleaned dataset to {CLEAN_PATH}  (shape={df.shape})")

# ============================================================================
# 6. Group-level summary table
# ============================================================================
print("\n" + "=" * 70)
print("GROUP SUMMARY (aggregated by channel)")
print("=" * 70)

agg = df.groupby(GROUP_COL).agg(
    n_campaigns=("Campaign_ID", "count"),
    total_impressions=("Impressions", "sum"),
    total_clicks=("Clicks", "sum"),
    total_conversions=("Conversions", "sum"),
    total_cost=("Acquisition_Cost", "sum"),
    total_revenue=("Revenue", "sum"),
)
agg["CTR"] = agg["total_clicks"] / agg["total_impressions"]
agg["Conversion_Rate"] = agg["total_conversions"] / agg["total_clicks"]
agg["CPA"] = agg["total_cost"] / agg["total_conversions"]
agg["ROAS"] = agg["total_revenue"] / agg["total_cost"]
agg["Profit"] = agg["total_revenue"] - agg["total_cost"]
agg["Profit_Margin"] = agg["Profit"] / agg["total_revenue"]
agg = agg.replace([np.inf, -np.inf], np.nan).round(4)

print(agg.to_string())
agg.to_csv("outputs/group_summary.csv")
print("\nSaved outputs/group_summary.csv")

# ============================================================================
# 7. Visualize group performance overview
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

metrics_to_plot = [
    ("CPA", "CPA by Channel ($)", "lower is better"),
    ("ROAS", "ROAS by Channel (x)", "higher is better"),
    ("Conversion_Rate", "Conversion Rate by Channel", "higher is better"),
    ("total_conversions", "Total Conversions by Channel", ""),
    ("total_cost", "Total Cost by Channel ($)", ""),
    ("Profit", "Total Profit by Channel ($)", ""),
]

for ax, (col, title, note) in zip(axes, metrics_to_plot):
    sorted_agg = agg[col].sort_values()
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in sorted_agg]
    ax.barh(sorted_agg.index, sorted_agg.values, color=colors)
    ax.set_title(f"{title}\n({note})" if note else title, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("outputs/group_metrics_overview.png", dpi=150)
print("Saved outputs/group_metrics_overview.png")
plt.close()

# ============================================================================
# 8. Distribution plots (variability within each channel)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

groups = df[GROUP_COL].unique()

# Histogram overlay of CPA
for g in groups:
    vals = df.loc[df[GROUP_COL] == g, "CPA"].dropna()
    axes[0].hist(vals, bins=40, alpha=0.4, label=g)
axes[0].set_title("CPA Distribution by Channel")
axes[0].set_xlabel("CPA ($)")
axes[0].legend(fontsize=7)

# Boxplot of ROI
df.boxplot(column="ROI", by=GROUP_COL, ax=axes[1], rot=45)
axes[1].set_title("ROI Distribution by Channel")
axes[1].set_xlabel("")

# Boxplot of Conversion_Rate
df.boxplot(column="Conversion_Rate", by=GROUP_COL, ax=axes[2], rot=45)
axes[2].set_title("Conversion Rate Distribution by Channel")
axes[2].set_xlabel("")

plt.suptitle("")
plt.tight_layout()
plt.savefig("outputs/group_distributions.png", dpi=150)
print("Saved outputs/group_distributions.png")
plt.close()

print("\n" + "=" * 70)
print("STEP 2 & 3 COMPLETE")
print("=" * 70)
print(f"Groups identified: {list(agg.index)}")
print(f"Sample sizes per group:\n{df[GROUP_COL].value_counts()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
