# Marketing Channel Statistical Analysis Lab

## Overview

This project analyzes the performance of six marketing channels using statistical methods to determine whether observed differences in key marketing metrics are statistically and practically meaningful. The analysis focuses on CPA, conversion rates, ROAS, multiple-comparison correction, confidence intervals, statistical power, and a final business-oriented budget recommendation.

## Dataset Documentation

- **Dataset name:** Marketing Campaign Performance Dataset
- **Source:** Kaggle
- **URL:** [Marketing Campaign Performance Dataset](https://www.kaggle.com/datasets/manishabhatt22/marketing-campaign-performance-dataset?utm_source=chatgpt.com)

### Why This Dataset Was Chosen

The dataset contains multi-channel campaign performance data with distinct channel groupings and key raw metrics—including impressions, clicks, conversions, spend, and revenue—required to calculate CPA, ROAS, and conversion rates across multiple groups for statistical testing.

## Initial Exploratory Insights

- **CPA (lower is better):** Website had the lowest observed CPA at approximately $281.93.
- **ROAS (higher is better):** Facebook had the highest observed ROAS at approximately 5.03x.
- **Conversion rate:** Email had the highest observed conversion rate at approximately 8.03%.
- **Variability:** Aggregate channel-level metrics were extremely close, with no obvious large outliers.
- **Initial pattern:** CPA varied by only about $3 between the best and worst-performing channels, motivating formal statistical testing to determine whether these small differences represented meaningful effects or random variation.

## How to Run

### 1. Install Dependencies

Ensure Python 3.8 or later is installed, then install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scipy
```

### 2. Run the Analysis Scripts

Run the scripts sequentially:

```bash
python data_exploration.py
python statistical_analysis.py
python power_analysis.py
python business_recommendations.py
```

## Repository Structure

```text
.
├── data/
│   └── marketing_campaign_dataset.csv      # Raw source dataset
├── outputs/                                # Processed data, results, and visualizations
│   ├── marketing_data.csv                  # Primary processed dataset
│   ├── group_summary.csv                   # Summary metrics by marketing channel
│   ├── cpa_comparison_results.csv          # Pairwise CPA test results
│   ├── fisher_comparison_results.csv       # Pairwise conversion-rate test results
│   ├── cpa_confidence_intervals.csv        # Bootstrap 95% confidence intervals for CPA
│   ├── cpa_confidence_intervals.png        # Confidence interval visualization
│   ├── budget_allocation.csv               # Final $500,000 monthly budget allocation
│   ├── correction_summary.csv              # Multiple-comparison correction summary
│   ├── power_analysis_cpa.png              # Power curves
│   ├── power_analysis_results.csv          # Power analysis outputs
│   ├── min_sample_size_cpa.csv             # Minimum sample-size estimates
│   ├── power_adequacy_conversion_rate.csv  # Conversion-rate power adequacy results
│   ├── fdr_significant_findings.txt        # Significant findings after FDR correction
│   ├── qualitative_prioritization.txt      # Qualitative ranking summary
│   ├── metric_comparison_heatmap.png       # CPA comparison heatmap
│   ├── group_metrics_overview.png          # Channel metric overview chart
│   ├── group_distributions.png             # Group distribution plots
│   ├── rate_comparison.png                 # Rate comparison visualization
│   ├── correction_comparison.png           # Uncorrected vs. FDR comparison
│   └── budget_allocation.png               # Recommended budget allocation chart
├── data_exploration.py                     # Data loading, cleaning, and exploratory summaries
├── statistical_analysis.py                 # Hypothesis tests and FDR corrections
├── power_analysis.py                       # Empirical power simulations and sample-size analysis
├── business_recommendations.py             # Confidence intervals, rankings, and budget allocation
├── executive_memo.md                       # Business recommendations and executive summary
├── reflection.md                           # Reflection on findings and analytical limitations
└── README.md                               # Project documentation and repository guide
```

## Analysis Assumptions

- **Data path:** Raw source data is located in `data/`, and all processed datasets, generated plots, and text logs are exported to `outputs/`.
- **Reproducibility:** Simulations and bootstrap samples use fixed random seeds (`np.random.default_rng(42)`).
- **Statistical significance threshold:** Hypothesis tests and power calculations use `α = 0.05`.
- **Multiple comparisons:** The Benjamini-Hochberg False Discovery Rate (FDR) procedure is used to account for false-positive risk when conducting multiple simultaneous comparisons.
- **Channel Grouping:** While the general scenario models 7 default channels, the Kaggle dataset contains 6 distinct, well-populated marketing channels (e.g., Facebook, Email, Website, Search, Video, Influencer), providing ample statistical power for all 15 pairwise comparisons.
