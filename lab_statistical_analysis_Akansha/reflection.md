# Reflection

**What surprised you about the results?**
The most surprising outcome was how identical performance across all six marketing channels proved to be. Initial point estimates showed minor variations—such as Facebook leading on ROAS or Website showing the lowest CPA—which typically invites teams to reallocate budget immediately. However, underlying statistical testing revealed that these gaps are virtually non-existent, serving as a powerful reminder that raw averages can be highly misleading without hypothesis testing.

**How did multiple comparisons correction change your conclusions?**
Applying multiple testing corrections fundamentally shifted the narrative from "some channels are winning" to "all channels perform equally". Without corrections, standard $p$-values flagged 9 out of 15 conversion rate pairs as statistically significant. Once we applied the Benjamini-Hochberg False Discovery Rate (FDR) procedure, we accounted for the increased chance of false positives inherent in running 30 simultaneous tests, proving that none of the CPA differences were real and that the conversion rate gaps were practically negligible.

**What are the limitations of this analysis?**
* **Lack of Granular Segmentation:** This analysis evaluates aggregate, channel-level performance without controlling for customer segments, geographic regions, or specific campaign creatives available in the raw data.
* **Over-reliance on Aggregated Proportions:** Extremely large sample sizes (hundreds of thousands of daily attempts) drive down standard errors, causing micro-differences (0.02 to 0.05 percentage points) to trigger statistical significance despite having zero commercial impact.
* **Attribution Scope:** The data reflects last-click style performance metrics without measuring upper-funnel influence or cross-channel assist effects.

**How would you communicate these findings to non-technical stakeholders?**
I would frame the result as a strategic victory rather than a statistical failure: "We have good news—none of our channels are failing, and none are runaway winners". I would explain that while big data tools can detect tiny fractional differences in conversion rates, those gaps translate to pennies in the real world. Instead of gambling budget on statistical noise, our best move is to keep spend evenly diversified across platforms to avoid over-relying on a single channel, while choosing where to invest based on platform creative costs and audience reach.