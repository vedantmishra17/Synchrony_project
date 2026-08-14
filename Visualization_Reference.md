# Visualization Reference for the XYZ-ABC SoW Project

This note translates the categorized chart list into a practical recommendation set for this pipeline.

## Already Covered

These chart types are already represented in the current work or are effectively covered by existing outputs:

- Histogram
- Bubble chart
- Elbow and silhouette plots
- Calibration curve
- SHAP summary / feature driver view

## Highest-Value Additions

These are the chart types that would add the most value to the final deck without duplicating what we already have:

- Box plot of `SoW_lifetime` by segment
- Diverging bar chart of `SoW_H2_minus_H1` by segment
- Stacked bar chart of payment-method mix by segment
- 100% stacked bar chart of category mix by segment
- Line chart of population-level monthly SoW over time
- ROC curve and precision-recall curve for the decline classifier
- Lift / gains chart for top-decile targeting
- PCA scatter of measurable customers colored by cluster
- Cluster centroid heatmap for segment interpretation

## Best Fits By Analytical Purpose

### Distribution

- Histogram: use for `monetary`, `tenure_days`, and `recency_days`.
- Box plot: best for comparing `SoW_lifetime` or `monetary` across segments.
- Violin plot: useful when the distribution has a near-zero spike and a long tail.
- KDE / density plot: best for comparing Prime vs Non-Prime SoW distributions.
- ECDF: useful for statements like “what percent of customers are below a given SoW threshold?”

### Relationship

- Scatter plot: good for `tenure_days` vs `SoW_lifetime` or `frequency` vs `monetary`.
- Correlation heatmap: strong diagnostic slide for the Step 4 feature table.
- Pair plot: useful for EDA, but usually too dense for presentation.

### Comparison

- Grouped bar chart: best for comparing average SoW by segment.
- Stacked bar chart: best for category or payment mix by segment.
- Diverging bar chart: best for showing positive vs negative `SoW_H2_minus_H1`.
- Radar chart: useful only if kept to a small number of segments and metrics.

### Composition

- Donut chart: acceptable for a simple segment size summary.
- 100% stacked bar: better than a pie chart when comparing composition across segments.
- Treemap: good for hierarchical segment sizing.

### Time Series

- Line chart: best for monthly population-level SoW trend.
- Area chart: useful for ABC spend vs total spend over time.
- Cohort heatmap: advanced option if there is time to build it.

### Model Evaluation

- ROC curve: stronger visual than AUC alone.
- Precision-recall curve: especially useful because the use case is targeted outreach.
- Confusion matrix: useful at the chosen classification threshold.
- Lift chart: directly answers the marketing question.

### Segmentation

- PCA scatter: shows how well the measurable customers separate in 2D.
- Cluster centroid heatmap: best compact summary of cluster profiles.

## Recommendation

If time is limited, the best sequence of additions is:

1. Segment profile box / bar charts
2. Diverging trend bar chart
3. ROC and precision-recall curves
4. Lift chart
5. PCA cluster visualization

That gives the deck a balanced story: distribution, behavior, model quality, and segment separation.
