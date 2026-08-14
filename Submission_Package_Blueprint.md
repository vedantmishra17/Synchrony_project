# Synchrony Analytics Hackathon 2026
## Submission Package Blueprint

This document turns the completed `Credit_card_sow.py` pipeline into a judge-ready submission plan for both the technical report and the executive presentation. It is grounded in the actual outputs produced in the workspace:

- `Customer_Feature_Table.csv`: 38,164 customers, 35 columns
- `Customer_Segment_Profile.csv`: 8 segments
- `Customer_Segment_Strategy.csv`: 8 segments
- `Customer_Segment_and_Risk.csv`: 38,164 scored customers
- `KMeans_Model_Selection.csv`: k = 4, 5, 6 evaluation, selected k = 6
- `Decline_Model_Metrics.json`: logistic AUC 0.642, XGBoost AUC 0.685
- `Decline_Model_Calibration.csv`: 10 calibration bins
- Visuals in `visualizations/`: 5 executive charts

---

# 1. Complete Report Outline

## 1. Executive Summary

### Purpose
Give judges a fast, decision-oriented summary of the problem, the methodology, the main findings, and the final business actions.

### Key Content
- Business problem: customers still shop at XYZ Inc. but increasingly pay with competing methods instead of the ABC Bank co-branded card.
- Analytical approach: clean the transaction universe, measure lifetime SoW, create half-window trend signals, build a customer ABT, segment the base, and score decline risk.
- Headline results:
  - 38,164 customers scored in the final ABT
  - 8 customer segments
  - 3 operating populations: Measurable-SoW, Dormant Cardholders, Return-Heavy/Anomalous
  - Logistic model AUC = 0.642
  - XGBoost model AUC = 0.685
  - Top-decile precision = 0.685 for XGBoost
- Core recommendation: defend high-value ABC loyalists, win back cash/UPI drifters, recapture cross-card electronics spend, and route return-heavy cases to service review.

### Charts/Tables to Include
- One executive summary figure showing the segment landscape
- One small KPI table with customer count, segment count, and model AUCs

### Important Findings
- Lifetime SoW is the primary metric, not the monthly ratio.
- Out-of-bounds SoW rows are retained and flagged rather than clipped.
- Decline risk is concentrated in the largest high-value segments.

### Business Interpretation
The pipeline identifies where ABC is losing wallet share, which customers are most at risk, and what intervention is appropriate for each behavioral group.

---

## 2. Business Context

### Purpose
Explain why Share of Wallet matters for card economics and why declining wallet share is a strategic risk.

### Key Content
- ABC is not simply losing transactions; it is losing payment instrument preference.
- SoW affects interchange, loyalty economics, engagement, and future share capture.
- Declining SoW is an early warning signal for churn, competitor migration, or payment-method drift.

### Charts/Tables to Include
- Simple SoW framework diagram
- A conceptual wallet-share funnel or payment-method stack

### Important Findings
- The data shows customers still shop across the same merchant universe but route spend through cash, UPI, debit, and other bank cards.
- The highest-value segments are not always the highest-SoW segments.

### Business Interpretation
The problem is not purely acquisition. It is share retention within an existing customer base.

---

## 3. Research Foundation

### Purpose
Anchor the findings in prior literature and show that the methodology aligns with known customer behavior theory.

### Key Content
For each reference, explain:
- Core concept
- Why it matters here
- Whether the project supports it
- Which output demonstrates it

### Suggested Literature Integration
- **Du & Kamakura (2006)**: multi-homing behavior; customers use multiple payment instruments across the same shopping occasions.
- **Milkman et al. (2011)**: present bias and delayed reward discounting; immediate payment convenience can outrank delayed loyalty value.
- **McKinsey Loyalty Research (2020)**: commitment and membership value influence repeat behavior, but only when the value is salient and easy to realize.
- **Keiningham et al. (2007)**: loyalty and share of wallet are related but not identical; high satisfaction does not guarantee high SoW.
- **Wirtz et al. (2007)**: reward program design matters; incentives must be visible, timely, and behaviorally specific.
- **Cooil et al. (2007)**: wallet behavior changes longitudinally; trend matters more than a single snapshot.

### Charts/Tables to Include
- A literature-to-finding mapping table

### Important Findings
- The segment structure and payment mix support multi-homing.
- Trend analysis is justified because wallet share changes over time, not just in level.

### Business Interpretation
The project is not only operationally useful; it is also consistent with established customer behavior theory.

---

## 4. Data Understanding

### Purpose
Document the raw data, the schema, and the quality checks that shaped the analysis.

### Key Content
- Source files: `Transaction Data.csv`, `Customer Data.csv`, `Category Code.csv`, `Payment Code.csv`
- Full cleaned transaction history:
  - `Transaction_Data_Cleaned.csv`: 378,793 rows
  - `Transaction_Data_SoW_Eligible.csv`: 262,883 rows
- Customer master / ABT:
  - `Customer_Feature_Table.csv`: 38,164 rows, 35 columns
- Transaction date handling:
  - Transaction dates parsed as dates
  - Card dates used to define active card window
- Returns:
  - Return rows are signed negative
  - `Number_of_Transactions = 0` for return rows
- Fiscal year:
  - Aug = fiscal month 1
  - Jul = fiscal month 12

### Charts/Tables to Include
- Data schema table
- Missingness / validity summary
- Transaction counts before and after active-window filtering

### Important Findings
- 6,836 customers had no open date and were dropped from SoW analysis.
- Their 67,632 transaction rows were removed from the retained analysis population.

### Business Interpretation
The SoW population is the set of actual ABC cardholders with a valid active window, not everyone in the raw customer file.

---

## 5. Methodology

### Purpose
Explain the analytical pipeline end-to-end in business language.

### Key Content
Raw data -> cleaning -> SoW -> feature engineering -> segmentation -> prediction -> strategy mapping

### Detailed Flow
1. Merge transactions with category, payment, and customer lookup tables.
2. Clean dates, returns, and active-window eligibility.
3. Compute lifetime SoW and half-window SoW.
4. Build the customer-level ABT.
5. Segment the measurable population.
6. Score decline risk.
7. Map segments to actions.

### Charts/Tables to Include
- A pipeline flow diagram
- A table of intermediate output files

### Important Findings
- Lifetime SoW is the official metric.
- Monthly and quarterly panels are retained as diagnostics.
- Half-window delta replaces fragile long monthly slopes.

### Business Interpretation
The methodology is designed to match the case study definition and avoid artifacts from return-heavy aggregation.

---

## 6. Share of Wallet Methodology

### Purpose
Define the SoW logic precisely and explain why the project evolved toward lifetime SoW.

### Key Content
- `SoW_lifetime = ABC spend / total active-window spend`
- Guard: `total_spend <= 0 -> NaN`
- Active card filtering by open/close dates
- Return handling with signed net amounts
- Monthly and quarterly panels retained only for diagnostics
- Half-window trend:
  - `SoW_H2_minus_H1`
  - used as a direction feature, not as a noisy time series slope

### Charts/Tables to Include
- SoW definition diagram
- Box plot of SoW by segment
- Diverging bar chart of SoW delta by segment

### Important Findings
- Out-of-range SoW values are not clipped.
- Flagged anomalies are kept visible as business signals.

### Business Interpretation
The case study asks for a wallet share measure, not a monthly panel. Lifetime SoW is the cleanest answer.

---

## 7. Feature Engineering

### Purpose
Explain the customer ABT and the features used for segmentation and prediction.

### Key Content
- Behavioral:
  - tenure_days
  - recency_days
  - frequency
  - monetary
  - avg_ticket
- Category mix:
  - grocery_pct
  - electronics_pct
  - other_pct
- Payment mix:
  - payment_pct_abc_bank_credit_card
  - payment_pct_cash_upi
  - payment_pct_debit_card
  - payment_pct_other_bank_credit_card
  - payment_pct_xyz_wallet
- SoW features:
  - SoW_lifetime
  - SoW_H2_minus_H1
  - trend_status
- Flags:
  - has_measurable_sow
  - is_dormant_cardholder
  - is_abc_return_dominated
  - is_cross_channel_return_skewed

### Charts/Tables to Include
- Feature importance table
- Correlation heatmap
- RFM summary table

### Important Findings
- The final ABT keeps dormant customers.
- `monetary` equals lifetime total spend.
- `payment_pct_abc_bank_credit_card` is a strong explanatory feature.

### Business Interpretation
The feature table is the bridge between raw transactions and action-oriented customer strategy.

---

## 8. Customer Segmentation

### Purpose
Describe the segment solution and the business meaning of each group.

### Key Content
Segments in `Customer_Segment_Strategy.csv`:
- High-Value Mixed-Basket ABC Loyalists
- Core Mixed-Basket Cash/UPI Users
- Core Mixed-Basket ABC Loyalists
- Dormant Cardholders
- High-Value Electronics Cross-Card Users
- Value Mixed-Basket Debit-First Users
- Return-Heavy/Anomalous
- Value Grocery Cash Drifters

### Segment Highlights
- High-Value Mixed-Basket ABC Loyalists:
  - 15,037 customers
  - avg_sow_lifetime = 0.3298
  - flagged_share = 0.0248
- Core Mixed-Basket ABC Loyalists:
  - 4,476 customers
  - avg_sow_lifetime = 0.7463
  - declining_share = 0.2978
- Core Mixed-Basket Cash/UPI Users:
  - 5,913 customers
  - avg_sow_lifetime = 0.0864
  - flagged_share = 0.1072
- High-Value Electronics Cross-Card Users:
  - 4,174 customers
  - avg_sow_lifetime = 0.3139
- Value Grocery Cash Drifters:
  - 964 customers
  - flagged_share = 0.2064
- Dormant Cardholders:
  - 4,450 customers
  - no measurable SoW
- Return-Heavy/Anomalous:
  - 1,230 customers
  - no bounded SoW average

### Charts/Tables to Include
- Segment profile table
- Box plot of SoW by segment
- Cluster centroid heatmap

### Important Findings
- K = 6 selected on the measurable population.
- The silhouette score is modest but improves at k=6.
- Two rule-based groups are handled outside the K-Means solution.

### Business Interpretation
The segmentation is not just descriptive; it separates retention, win-back, activation, behavior-shift, and service-review populations.

---

## 9. Predictive Modeling

### Purpose
Document the decline-risk model and its quality.

### Key Content
- Training population: 22,011 customers with determined trend labels
- Train/test split:
  - 17,608 train rows
  - 4,403 test rows
- Models:
  - Logistic Regression
  - XGBoost
- Metrics:
  - Logistic AUC = 0.642
  - Logistic average precision = 0.630
  - Logistic top-decile precision = 0.660
  - XGBoost AUC = 0.685
  - XGBoost average precision = 0.654
  - XGBoost top-decile precision = 0.685
  - XGBoost Brier score = 0.201
- Calibration:
  - 10-bin calibration table shows reasonably aligned observed vs predicted decline rates
- SHAP:
  - Driver summary provided for customer-level scores

### Charts/Tables to Include
- ROC curve
- Precision-recall curve
- Calibration curve
- SHAP summary plot
- Coefficient table

### Important Findings
- XGBoost outperforms logistic regression.
- Top-decile precision is materially better than random outreach.
- The model is useful for prioritization, not absolute certainty.

### Business Interpretation
The model is good enough to rank customers for action, especially when combined with segment context.

---

## 10. Strategic Recommendations

### Purpose
Translate the analysis into customer actions.

### Key Content
- High-Value Mixed-Basket ABC Loyalists:
  - retention offers
  - bonus cashback on grocery/electronics
  - Prime upgrade nudge
- Core Mixed-Basket Cash/UPI Users:
  - win-back campaign
  - no annual fee reminder
  - category reward emphasis
- Core Mixed-Basket ABC Loyalists:
  - loyalty defense
  - milestone-based reward framing
- Dormant Cardholders:
  - activation offer
- High-Value Electronics Cross-Card Users:
  - electronics cashback recapture
- Value Mixed-Basket Debit-First Users:
  - payment-switch nudge
  - friction-reduction incentive
- Return-Heavy/Anomalous:
  - service review / operational follow-up
- Value Grocery Cash Drifters:
  - grocery threshold bonus

### Charts/Tables to Include
- Segment strategy table
- Offer matrix by segment

### Important Findings
- The largest opportunities are not just the biggest customers; they are the biggest customers with declining share or payment drift.

### Business Interpretation
The right action depends on segment behavior, not a single blanket incentive.

---

## 11. Limitations

### Purpose
State the methodological caveats honestly.

### Key Content
- Weak cluster separation
- Synthetic return anomaly
- Moderate predictive power
- Diagnostic monthly/quarterly SoW not used directly for modeling
- Remaining redundancy in the logistic feature set

### Charts/Tables to Include
- A short limitations callout box

### Important Findings
- Silhouette values are low.
- Some SoW values remain outside [0, 1] due to return behavior.

### Business Interpretation
The solution is strong for a hackathon submission, but it is still a decision-support tool, not a production credit policy engine.

---

## 12. Conclusion

### Purpose
Close the report with the value proposition and next steps.

### Key Content
- What was learned
- Why it matters
- What ABC should do next

### Charts/Tables to Include
- Final summary table

### Important Findings
- Wallet share can be measured, segmented, and operationalized.

### Business Interpretation
The analysis gives ABC a practical retention, reactivation, and recapture framework.

---

# 2. Complete PPT Outline

## Slide 1 - Title & Executive Summary

### Objective
Set the business context and present the headline outcome.

### Bullet Points
- Declining Share of Wallet in ABC Bank co-branded credit-card customers
- 38,164 customers scored
- 8 customer segments
- XGBoost AUC 0.685
- Top-decile precision 0.685

### Speaker Notes
Open with the business risk: customers still buy, but not always with the ABC card.

### Visual Recommendation
One-page executive summary with segment icons and KPI tiles.

### Supporting Research Citation
Keiningham et al. (2007); Cooil et al. (2007)

---

## Slide 2 - Business Problem

### Objective
Explain the commercial problem in plain language.

### Bullet Points
- Customers are multi-homing across payment methods
- ABC is losing payment preference, not necessarily merchant demand
- Declining SoW threatens interchange and loyalty economics

### Speaker Notes
Frame the issue as share loss inside an existing customer base.

### Visual Recommendation
Payment-method funnel or wallet-share schematic.

### Supporting Research Citation
Du & Kamakura (2006); Milkman et al. (2011)

---

## Slide 3 - Data Overview

### Objective
Show the scope and quality of the data used.

### Bullet Points
- 446,425 raw transactions in the original transaction file
- 378,793 cleaned transaction rows after removing never-opened customers
- 262,883 active-window SoW-eligible rows
- 38,164 customer-level ABT rows

### Speaker Notes
Highlight that the analysis retains dormant cardholders in the ABT.

### Visual Recommendation
Simple data pipeline graphic with row counts.

### Supporting Research Citation
Cooil et al. (2007)

---

## Slide 4 - Methodology

### Objective
Show the analytical flow from raw data to strategy.

### Bullet Points
- Merge
- Clean
- SoW
- Feature engineering
- Segmentation
- Prediction
- Strategy mapping

### Speaker Notes
Emphasize that the design follows the case-study rules, especially active-window filtering.

### Visual Recommendation
Pipeline flow chart.

### Supporting Research Citation
Du & Kamakura (2006)

---

## Slide 5 - SoW Framework

### Objective
Explain the corrected SoW definition.

### Bullet Points
- Lifetime SoW is the primary metric
- Guard: total_spend <= 0 -> NaN
- Half-window delta is used for trend
- Monthly and quarterly panels are diagnostics only

### Speaker Notes
Explain why the project moved away from monthly ratios.

### Visual Recommendation
Box plot of lifetime SoW by segment and diverging SoW delta chart.

### Supporting Research Citation
Cooil et al. (2007); Keiningham et al. (2007)

---

## Slide 6 - Customer Segments

### Objective
Show how the base divides into actionable groups.

### Bullet Points
- 8 segments
- High-value loyalists
- Cash/UPI drift
- Electronics cross-card behavior
- Dormant accounts
- Return-heavy anomaly group

### Speaker Notes
Point out that not all segments are marketing targets; some are service or activation targets.

### Visual Recommendation
Cluster profile heatmap or segment summary table.

### Supporting Research Citation
Du & Kamakura (2006)

---

## Slide 7 - Declining Wallet Analysis

### Objective
Show where decline is concentrated and how trend differs by group.

### Bullet Points
- Decline is concentrated in large high-value groups
- Half-window delta reveals directionality
- Some segments have strong SoW but still meaningful decline share

### Speaker Notes
This is where the story shifts from segmentation to wallet erosion.

### Visual Recommendation
Diverging SoW delta bar chart.

### Supporting Research Citation
Cooil et al. (2007)

---

## Slide 8 - Predictive Modeling

### Objective
Explain the decline-risk model and its performance.

### Bullet Points
- Logistic regression and XGBoost compared
- XGBoost selected for risk ranking
- AUC 0.685
- Top-decile precision 0.685

### Speaker Notes
Present the model as a prioritization engine rather than a perfect classifier.

### Visual Recommendation
ROC curve and precision-recall curve.

### Supporting Research Citation
Milkman et al. (2011); Wirtz et al. (2007)

---

## Slide 9 - Key Insights

### Objective
Summarize the most important findings in plain English.

### Bullet Points
- Loyalty does not equal SoW
- Cash/UPI drift is a major threat
- Electronics is a strong recapture opportunity
- Return-heavy cases are operationally distinct

### Speaker Notes
Tie behavior patterns to segment stories.

### Visual Recommendation
Top findings summary slide with 3-4 tiles.

### Supporting Research Citation
Keiningham et al. (2007); McKinsey Loyalty Research (2020)

---

## Slide 10 - Strategy Recommendations

### Objective
Translate findings into actions.

### Bullet Points
- Retain loyalists
- Win back cash/UPI users
- Capture electronics spend
- Activate dormant cardholders
- Route anomalies to service review

### Speaker Notes
Show that each segment gets a tailored intervention.

### Visual Recommendation
Segment-to-offer matrix.

### Supporting Research Citation
Wirtz et al. (2007); McKinsey Loyalty Research (2020)

---

## Slide 11 - Expected Impact

### Objective
Explain the expected business value.

### Bullet Points
- Better targeting efficiency
- More relevant offers
- Higher ABC card usage
- Earlier intervention on decline

### Speaker Notes
Quantify impact qualitatively if not directly measured.

### Visual Recommendation
Lift chart plus action funnel.

### Supporting Research Citation
Milkman et al. (2011)

---

## Slide 12 - Conclusion

### Objective
Close with the business outcome and next steps.

### Bullet Points
- The problem is measurable
- The customers are identifiable
- The interventions are segment-specific
- The next step is deployment

### Speaker Notes
End with a direct recommendation: use the segment strategy table as the operating plan.

### Visual Recommendation
Closing slide with the final strategy table headline.

### Supporting Research Citation
Cooil et al. (2007)

---

# 3. Recommended Charts

Use the five visualizations already generated in `visualizations/`:

1. `01_segment_profile_boxplot.png`
2. `02_sow_delta_diverging_bar.png`
3. `03_model_performance_roc_pr.png`
4. `04_lift_gains_chart.png`
5. `05_pca_scatter_cluster_heatmap.png`

Additional supporting charts if you want to expand the deck:

- Calibration curve from `Decline_Model_Calibration.csv`
- Segment strategy table as a formatted slide
- Segment size bar chart
- Payment-method mix stacked bar chart

---

# 4. Recommended Tables

1. Output inventory table
2. Data quality / cleaning table
3. Segment summary table
4. Strategy mapping table
5. Model metrics table
6. Literature-to-finding mapping table

Suggested actual content:
- Segment counts and SoW averages from `Customer_Segment_Profile.csv`
- Model metrics from `Decline_Model_Metrics.json`
- Decile calibration from `Decline_Model_Calibration.csv`
- Segment action map from `Customer_Segment_Strategy.csv`

---

# 5. Academic Literature Integration

## Du & Kamakura (2006)
- Core idea: customers multi-home across providers and payment instruments.
- Why relevant: the same customer can keep shopping at XYZ but pay with a different card or method.
- Supported by findings: yes.
- Evidence:
  - payment mix features in the ABT
  - segments such as Cash/UPI users and Debit-First users
  - High-Value Electronics Cross-Card Users

## Milkman et al. (2011)
- Core idea: present bias affects reward uptake; immediate convenience often beats delayed benefits.
- Why relevant: reward value may be discounted if the friction is too high or the reward is not immediate.
- Supported by findings: partially and directionally.
- Evidence:
  - cash/UPI drift segments
  - need for point-of-sale reminders and friction-reduction nudges

## McKinsey Loyalty Research (2020)
- Core idea: commitment and membership value matter when the benefit is visible and easy to realize.
- Why relevant: Prime and prime-like customers may respond differently from non-prime customers.
- Supported by findings: yes, through segment differences in Prime share and loyalty defense needs.
- Evidence:
  - Prime share is high across several high-value segments
  - Loyalty Defense recommendation for Core Mixed-Basket ABC Loyalists

## Keiningham et al. (2007)
- Core idea: loyalty and share of wallet are related but not identical.
- Why relevant: high satisfaction or high value does not guarantee full wallet capture.
- Supported by findings: yes.
- Evidence:
  - high-value segments still show declining share
  - SoW_lifetime varies materially inside high-value groups

## Wirtz et al. (2007)
- Core idea: reward programs work when rewards are specific, credible, and behavior-linked.
- Why relevant: the offer must match the behavior gap.
- Supported by findings: yes.
- Evidence:
  - grocery threshold bonuses for grocery drifters
  - electronics cashback for cross-card users
  - win-back offers for cash/UPI drift

## Cooil et al. (2007)
- Core idea: wallet behavior must be tracked longitudinally.
- Why relevant: one SoW snapshot misses the direction of change.
- Supported by findings: strongly yes.
- Evidence:
  - SoW_H2_minus_H1
  - trend_status
  - decline-risk model trained on trend-aware labels

---

# 6. Judge-Facing Storyline

## Problem
ABC’s customers are still active shoppers, but they are shifting spend away from the ABC co-branded card and into competing payment methods.

## Discovery
The analysis shows that wallet share is concentrated, uneven, and segment-specific. A small number of behavioral groups account for most of the actionable decline.

## Why It Is Happening
The evidence points to payment-method drift, category-specific competition, reward-friction mismatch, and separate treatment for dormant and return-heavy customers.

## Who Is Responsible
The main at-risk groups are:
- High-Value Mixed-Basket ABC Loyalists
- Core Mixed-Basket Cash/UPI Users
- High-Value Electronics Cross-Card Users
- Value Mixed-Basket Debit-First Users
- Value Grocery Cash Drifters

## How ABC Can Intervene
Use segment-specific offers:
- retention
- win-back
- loyalty defense
- activation
- cross-card recapture
- behavior-shift nudges
- service review for anomalous returns

## Expected Impact
The strategy should improve targeting efficiency, lift ABC card usage, and focus spend on customers where the wallet-share loss is most economically meaningful.

---

# 7. Executive Summary Draft

ABC Bank’s co-branded credit-card challenge is not simply customer inactivity. The core issue is declining share of wallet: customers continue shopping at XYZ Inc. but increasingly pay through competing methods. Using the completed `Credit_card_sow.py` pipeline, we cleaned the transaction universe, measured lifetime SoW over the active card window, engineered customer-level behavioral features, segmented the base into eight actionable groups, and scored decline risk for 38,164 customers.

The analysis shows that wallet erosion is concentrated in identifiable segments. High-value loyalists remain strategically important, but several groups are drifting toward cash, UPI, debit, or competing cards. XGBoost delivered the strongest decline-ranking performance with ROC AUC 0.685 and top-decile precision 0.685, making it suitable for prioritizing outreach. The final recommendation is a segmented action plan: retain high-value loyalists, win back cash/UPI users, recapture electronics spend, activate dormant cardholders, and route return-heavy anomalies to operational review.

---

# 8. Final Submission Recommendations

1. Use the PDF report as the technical appendix.
2. Use the PPT as the judge-facing narrative.
3. Lead with lifetime SoW, not the monthly panel.
4. Keep the monthly and quarterly tables as diagnostics only.
5. Put the five generated visuals into the presentation in the same order as the storyline.
6. Make the strategy table the final slide or appendix artifact.
7. Keep the model caveats honest:
   - cluster separation is weak
   - predictive power is moderate, not perfect
   - return anomalies are flagged, not hidden
8. Use the literature section to show that the findings are behaviorally grounded, not just statistically convenient.

---

# 9. Output Checklist

- `Credit_card_sow.py`
- `requirements.txt`
- `Synchrony_Analytics_Hackathon_Final_Report.pdf`
- `Synchrony_Analytics_Hackathon_Final_Report.md`
- `visualization_suite.py`
- `visualizations/01_segment_profile_boxplot.png`
- `visualizations/02_sow_delta_diverging_bar.png`
- `visualizations/03_model_performance_roc_pr.png`
- `visualizations/04_lift_gains_chart.png`
- `visualizations/05_pca_scatter_cluster_heatmap.png`
- `Customer_Segment_Strategy.csv`
- `Customer_Segment_Profile.csv`
- `Customer_Feature_Table.csv`
- `Decline_Model_Metrics.json`
- `Decline_Model_Calibration.csv`
- `Decline_Logistic_Coefficients.csv`

