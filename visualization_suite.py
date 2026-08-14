"""Executive-ready visualization suite for the SoW and segmentation pipeline.

This module is intentionally modular:
- Each figure has its own function.
- The functions accept a DataFrame and save a finished PNG.
- Column names are resolved through a small alias layer so the script works with
  both the prompt's ABT naming and the current workspace naming.

The five figures produced are:
1. Segment profile box plot
2. Diverging SoW trend bar chart
3. ROC and precision-recall curves
4. Lift and gains chart
5. PCA scatter plus cluster centroid heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


sns.set_theme(style="whitegrid")


COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "customer_id": ("Customer_ID", "customer_id"),
    "segment_name": ("Segment_Name", "cluster_segment", "Segment"),
    "cluster_id": ("Cluster_ID", "cluster_id"),
    "share_of_wallet": ("Share_of_Wallet", "SoW_lifetime", "Size_of_Wallet", "SoW"),
    "sow_delta": ("SoW_Delta", "SoW_H2_minus_H1"),
    "pca_1": ("PCA_Dim1", "PC1", "PCA1"),
    "pca_2": ("PCA_Dim2", "PC2", "PCA2"),
    "target": ("Is_Target_Platinum", "Is_Target", "trend_status"),
    "score": ("Pred_Prob_Platinum", "decline_risk_score", "Predicted_Probability"),
}


def resolve_column(df: pd.DataFrame, logical_name: str) -> str:
    """Return the first matching physical column for a logical field name."""
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"Could not find a column for '{logical_name}'. Tried: {list(COLUMN_ALIASES[logical_name])}"
    )


def ensure_output_dir(output_dir: Path | str) -> Path:
    """Create the output directory if needed and return it as a Path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_abt(abt_path: Path | str) -> pd.DataFrame:
    """Load the analytical base table from CSV."""
    return pd.read_csv(abt_path)


def standardize_target_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Convert the target column into a clean binary 0/1 series."""
    target = df[target_col]
    if pd.api.types.is_numeric_dtype(target):
        return target.fillna(0).astype(int)

    mapped = target.astype(str).str.strip().str.lower()
    # If the target is a textual trend label, treat Declining as the positive class.
    if mapped.isin({"declining", "1", "true", "yes"}).any():
        return mapped.isin({"declining", "1", "true", "yes"}).astype(int)

    return pd.to_numeric(target, errors="coerce").fillna(0).astype(int)


def prepare_visual_frame(abt: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the ABT with normalized column aliases where needed."""
    frame = abt.copy()

    alias_map = {
        "Segment_Name": resolve_column(frame, "segment_name"),
        "Cluster_ID": resolve_column(frame, "cluster_id"),
        "Share_of_Wallet": resolve_column(frame, "share_of_wallet"),
        "SoW_Delta": resolve_column(frame, "sow_delta") if any(
            col in frame.columns for col in COLUMN_ALIASES["sow_delta"]
        ) else None,
        "PCA_Dim1": resolve_column(frame, "pca_1") if any(
            col in frame.columns for col in COLUMN_ALIASES["pca_1"]
        ) else None,
        "PCA_Dim2": resolve_column(frame, "pca_2") if any(
            col in frame.columns for col in COLUMN_ALIASES["pca_2"]
        ) else None,
        "Target": resolve_column(frame, "target"),
        "Score": resolve_column(frame, "score"),
    }

    for canonical, source in alias_map.items():
        if source is not None and canonical not in frame.columns:
            frame[canonical] = frame[source]

    return frame


def discrete_palette(n: int, cmap: str = "viridis") -> List:
    """Create a discrete palette from a Matplotlib colormap."""
    palette = sns.color_palette(cmap, n_colors=max(n, 3))
    return palette[:n]


def plot_segment_profile_boxplot(
    abt: pd.DataFrame,
    output_path: Path | str,
    *,
    figsize: Tuple[int, int] = (14, 7),
) -> None:
    """Create a box plot with overlaid jittered points for SoW by segment."""
    frame = prepare_visual_frame(abt)
    segment_col = "Segment_Name"
    sow_col = "Share_of_Wallet"

    order = (
        frame.groupby(segment_col)[sow_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=frame,
        x=segment_col,
        y=sow_col,
        order=order,
        palette=discrete_palette(len(order), "viridis"),
        width=0.6,
        showfliers=True,
        hue=segment_col,
        dodge=False,
    )
    if ax.legend_:
        ax.legend_.remove()
    sns.stripplot(
        data=frame,
        x=segment_col,
        y=sow_col,
        order=order,
        color="black",
        alpha=0.25,
        size=2.5,
        jitter=0.22,
        ax=ax,
    )

    ax.set_title("Share of Wallet Distribution by Segment", fontweight="bold", pad=12)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Share of Wallet")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_diverging_sow_delta(
    abt: pd.DataFrame,
    output_path: Path | str,
    *,
    figsize: Tuple[int, int] = (14, 7),
) -> None:
    """Create a diverging bar chart of mean SoW change by segment."""
    frame = prepare_visual_frame(abt)
    segment_col = "Segment_Name"
    delta_col = "SoW_Delta"

    summary = (
        frame.groupby(segment_col, as_index=False)[delta_col]
        .mean()
        .sort_values(delta_col, ascending=True)
    )
    colors = np.where(summary[delta_col] >= 0, "#2ca02c", "#b22222")

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.barh(summary[segment_col], summary[delta_col], color=colors, edgecolor="none")
    ax.axvline(0, linestyle="--", color="gray", linewidth=1.5)

    for y, value in enumerate(summary[delta_col]):
        offset = 0.005 if value >= 0 else -0.005
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, y, f"{value:.3f}", va="center", ha=ha, fontsize=9)

    ax.set_title("Average SoW Delta by Segment (H2 - H1)", fontweight="bold", pad=12)
    ax.set_xlabel("Average SoW Delta")
    ax.set_ylabel("Segment")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_performance_curves(
    abt: pd.DataFrame,
    output_path: Path | str,
    *,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Plot ROC and precision-recall curves for the target classifier."""
    frame = prepare_visual_frame(abt)
    target_col = "Target"
    score_col = "Score"

    y_true = standardize_target_series(frame, target_col)
    y_score = pd.to_numeric(frame[score_col], errors="coerce")
    valid = y_true.notna() & y_score.notna()
    y_true = y_true.loc[valid]
    y_score = y_score.loc[valid]

    if y_true.nunique() < 2:
        raise ValueError("The target column must contain both positive and negative cases.")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    ap_score = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2.5, label=f"ROC AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Random baseline")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(recall, precision, color="#ff7f0e", linewidth=2.5, label=f"AP = {ap_score:.3f}")
    base_rate = float(y_true.mean())
    axes[1].axhline(base_rate, linestyle="--", color="gray", linewidth=1.5, label=f"Baseline = {base_rate:.3f}")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _compute_decile_frame(frame: pd.DataFrame, score_col: str, target_col: str) -> pd.DataFrame:
    """Return customer-level rows assigned to score deciles."""
    scored = frame[[score_col, target_col]].copy()
    scored = scored.dropna(subset=[score_col, target_col])
    scored = scored.sort_values(score_col, ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    scored["decile"] = pd.qcut(scored["rank"], 10, labels=False, duplicates="drop") + 1
    return scored


def plot_lift_and_gains_chart(
    abt: pd.DataFrame,
    output_path: Path | str,
    *,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Plot cumulative gains and lift for top-decile targeting."""
    frame = prepare_visual_frame(abt)
    target_col = "Target"
    score_col = "Score"

    y_true = standardize_target_series(frame, target_col)
    y_score = pd.to_numeric(frame[score_col], errors="coerce")
    valid = y_true.notna() & y_score.notna()
    scored = pd.DataFrame({target_col: y_true.loc[valid], score_col: y_score.loc[valid]})

    if scored[target_col].sum() == 0:
        raise ValueError("Lift/gains chart requires at least one positive target.")

    scored = scored.sort_values(score_col, ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    scored["target_cum"] = scored[target_col].cumsum()
    total_positive = scored[target_col].sum()
    scored["cum_gains"] = scored["target_cum"] / total_positive
    scored["cum_population"] = scored["rank"] / len(scored)
    scored["lift"] = scored["cum_gains"] / scored["cum_population"]

    deciles = (
        scored.assign(decile=lambda d: pd.qcut(d["rank"], 10, labels=False, duplicates="drop") + 1)
        .groupby("decile", as_index=False)
        .agg(
            cum_population=("cum_population", "max"),
            cum_gains=("cum_gains", "max"),
            lift=("lift", "max"),
        )
        .sort_values("decile")
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(
        deciles["cum_population"] * 100,
        deciles["cum_gains"] * 100,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Cumulative gains",
    )
    axes[0].plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1.5, label="Random baseline")
    axes[0].set_title("Cumulative Gains Chart", fontweight="bold")
    axes[0].set_xlabel("Customer Population Targeted (%)")
    axes[0].set_ylabel("High-Value Targets Captured (%)")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(
        deciles["cum_population"] * 100,
        deciles["lift"],
        marker="o",
        linewidth=2.5,
        color="#ff7f0e",
        label="Lift",
    )
    axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1.5, label="Random baseline")
    axes[1].set_title("Cumulative Lift Chart", fontweight="bold")
    axes[1].set_xlabel("Customer Population Targeted (%)")
    axes[1].set_ylabel("Lift over Random")
    axes[1].set_xlim(0, 100)
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    top_decile = deciles.iloc[0]["lift"] if not deciles.empty else np.nan
    if pd.notna(top_decile):
        axes[1].annotate(
            f"Top decile lift: {top_decile:.2f}x",
            xy=(deciles.iloc[0]["cum_population"] * 100, top_decile),
            xytext=(15, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def derive_cluster_feature_frame(
    abt: pd.DataFrame,
    cluster_features_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build or normalize the feature matrix used for the centroid heatmap."""
    if cluster_features_df is not None:
        features = cluster_features_df.copy()
        if "Cluster_ID" in features.columns:
            features = features.set_index("Cluster_ID")
        elif "cluster_id" in features.columns:
            features = features.set_index("cluster_id")
        return features

    frame = prepare_visual_frame(abt)
    cluster_col = "Cluster_ID"
    exclude = {
        "Customer_ID",
        "Segment_Name",
        "Cluster_ID",
        "Share_of_Wallet",
        "SoW_Delta",
        "Target",
        "Score",
        "PCA_Dim1",
        "PCA_Dim2",
    }
    candidate_cols = [
        c
        for c in frame.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not candidate_cols:
        raise ValueError("No numeric feature columns were found for the centroid heatmap.")

    numeric = frame[[cluster_col] + candidate_cols].copy()
    standardized = numeric.copy()
    for col in candidate_cols:
        col_mean = standardized[col].mean()
        col_std = standardized[col].std(ddof=0)
        if col_std == 0 or pd.isna(col_std):
            standardized[col] = 0.0
        else:
            standardized[col] = (standardized[col] - col_mean) / col_std

    cluster_features = standardized.groupby(cluster_col)[candidate_cols].mean()
    return cluster_features


def ensure_pca_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add PCA_Dim1 and PCA_Dim2 if the input does not already contain them."""
    if {"PCA_Dim1", "PCA_Dim2"}.issubset(frame.columns):
        return frame

    numeric_exclude = {
        "Customer_ID",
        "Cluster_ID",
        "Segment_Name",
        "Share_of_Wallet",
        "SoW_Delta",
        "Target",
        "Score",
        "PCA_Dim1",
        "PCA_Dim2",
    }
    feature_cols = [
        c
        for c in frame.columns
        if c not in numeric_exclude and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if len(feature_cols) < 2:
        raise ValueError("At least two numeric features are required to compute fallback PCA coordinates.")

    features = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = StandardScaler().fit_transform(features)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)

    enriched = frame.copy()
    enriched["PCA_Dim1"] = coords[:, 0]
    enriched["PCA_Dim2"] = coords[:, 1]
    return enriched


def plot_pca_scatter_and_heatmap(
    abt: pd.DataFrame,
    output_path: Path | str,
    *,
    cluster_features_df: Optional[pd.DataFrame] = None,
    max_heatmap_features: int = 16,
    figsize: Tuple[int, int] = (16, 7),
) -> None:
    """Plot PCA clusters and a cluster centroid heatmap in one figure."""
    frame = ensure_pca_coordinates(prepare_visual_frame(abt))
    pca1 = "PCA_Dim1"
    pca2 = "PCA_Dim2"
    cluster_col = "Cluster_ID"

    required = [pca1, pca2, cluster_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"PCA scatter requires these columns: {missing}")

    centroid_xy = (
        frame.groupby(cluster_col, as_index=False)[[pca1, pca2]]
        .mean()
        .sort_values(cluster_col)
    )

    cluster_features = derive_cluster_feature_frame(frame, cluster_features_df)
    # Keep the heatmap readable by using the most informative features.
    if cluster_features.shape[1] > max_heatmap_features:
        feature_spread = cluster_features.std(axis=0).sort_values(ascending=False)
        selected_features = feature_spread.head(max_heatmap_features).index.tolist()
        cluster_features = cluster_features[selected_features]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    cluster_ids = sorted(frame[cluster_col].dropna().unique().tolist())
    palette = discrete_palette(len(cluster_ids), "tab10")
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}

    for cid in cluster_ids:
        subset = frame[frame[cluster_col] == cid]
        axes[0].scatter(
            subset[pca1],
            subset[pca2],
            s=18,
            alpha=0.38,
            color=color_map[cid],
            label=f"Cluster {cid}",
            edgecolors="none",
        )

    axes[0].scatter(
        centroid_xy[pca1],
        centroid_xy[pca2],
        s=160,
        marker="X",
        color="black",
        label="Centroid",
        zorder=5,
    )
    for _, row in centroid_xy.iterrows():
        axes[0].annotate(
            str(int(row[cluster_col])),
            (row[pca1], row[pca2]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    axes[0].set_title("PCA Scatter by Cluster", fontweight="bold")
    axes[0].set_xlabel("PCA Dimension 1")
    axes[0].set_ylabel("PCA Dimension 2")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8, frameon=True)

    sns.heatmap(
        cluster_features,
        ax=axes[1],
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Standardized mean"},
    )
    axes[1].set_title("Cluster Centroid Heatmap", fontweight="bold")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Cluster")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_visual_suite(
    abt_path: Path | str = "Customer_Feature_Table.csv",
    output_dir: Path | str = "visualizations",
    *,
    segment_risk_path: Optional[Path | str] = None,
    cluster_features_path: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Generate all five visualizations and return their output paths."""
    out_dir = ensure_output_dir(output_dir)
    abt = load_abt(abt_path)

    # If the caller provides a separate cluster feature matrix, use it for the heatmap.
    cluster_features_df = pd.read_csv(cluster_features_path) if cluster_features_path else None

    # If the ABT does not already include the PCA or score fields, merge in the model outputs.
    if segment_risk_path is not None:
        segment_risk = pd.read_csv(segment_risk_path)
        merge_cols = [c for c in ["Customer_ID", "cluster_id", "cluster_segment", "decline_risk_score"] if c in segment_risk.columns]
        if "Customer_ID" in abt.columns and "Customer_ID" in segment_risk.columns:
            abt = abt.merge(segment_risk[merge_cols].drop_duplicates("Customer_ID"), on="Customer_ID", how="left")
            if "cluster_segment" in abt.columns and "Segment_Name" not in abt.columns:
                abt["Segment_Name"] = abt["cluster_segment"]
            if "cluster_id" in abt.columns and "Cluster_ID" not in abt.columns:
                abt["Cluster_ID"] = abt["cluster_id"]
            if "decline_risk_score" in abt.columns and "Pred_Prob_Platinum" not in abt.columns:
                abt["Pred_Prob_Platinum"] = abt["decline_risk_score"]

    outputs = {
        "segment_boxplot": out_dir / "01_segment_profile_boxplot.png",
        "diverging_delta": out_dir / "02_sow_delta_diverging_bar.png",
        "model_curves": out_dir / "03_model_performance_roc_pr.png",
        "lift_gains": out_dir / "04_lift_gains_chart.png",
        "pca_heatmap": out_dir / "05_pca_scatter_cluster_heatmap.png",
    }

    plot_segment_profile_boxplot(abt, outputs["segment_boxplot"])
    plot_diverging_sow_delta(abt, outputs["diverging_delta"])
    plot_model_performance_curves(abt, outputs["model_curves"])
    plot_lift_and_gains_chart(abt, outputs["lift_gains"])
    plot_pca_scatter_and_heatmap(
        abt,
        outputs["pca_heatmap"],
        cluster_features_df=cluster_features_df,
    )

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the visualization suite."""
    parser = argparse.ArgumentParser(description="Generate the five priority SoW/segmentation visuals.")
    parser.add_argument(
        "--abt",
        default="Customer_Feature_Table.csv",
        help="Path to the analytical base table CSV.",
    )
    parser.add_argument(
        "--segment-risk",
        default="Customer_Segment_and_Risk.csv",
        help="Optional path to customer-level segment/risk output.",
    )
    parser.add_argument(
        "--cluster-features",
        default=None,
        help="Optional path to a cluster feature matrix for the heatmap.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations",
        help="Directory where PNG files will be written.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    outputs = generate_visual_suite(
        abt_path=args.abt,
        output_dir=args.output_dir,
        segment_risk_path=args.segment_risk,
        cluster_features_path=args.cluster_features,
    )

    print("Generated visualizations:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
