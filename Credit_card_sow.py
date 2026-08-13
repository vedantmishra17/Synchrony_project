"""End-to-end customer analytics pipeline for the hackathon case study.

This script intentionally does a lot in one place because the project evolved step by step:

1. Read raw customer / transaction / lookup CSVs.
2. Clean and validate the raw data.
3. Build Share-of-Wallet (SoW) metrics at multiple grains.
4. Build a one-row-per-customer feature table.
5. Segment customers and score decline risk.
6. Produce strategy-ready output tables for downstream presentation work.

For a new teammate, the easiest way to understand the file is:

- Start with ``main()`` at the bottom to see the execution order.
- Then read the helper sections in order:
  validation -> SoW construction -> feature engineering -> segmentation -> modeling.

The comments below are intentionally dense so that someone onboarding to the project can
understand not just *what* each step does, but *why* that step exists in the business logic.
"""

import os
from pathlib import Path
import json
import re

# ``joblib`` sometimes tries to infer physical cores through a Windows-specific subprocess.
# In this environment that can emit noisy warnings, so we give it a sensible fallback.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import shap
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier


# All paths are relative to this file so the script can be run from any working directory.
BASE_DIR = Path(__file__).resolve().parent

# Raw input files from the case study.
TRANSACTION_DATA_PATH = BASE_DIR / "Transaction Data.csv"
CATEGORY_CODE_PATH = BASE_DIR / "Category Code.csv"
PAYMENT_CODE_PATH = BASE_DIR / "Payment Code.csv"
CUSTOMER_DATA_PATH = BASE_DIR / "Customer Data.csv"

# Intermediate and final analytical outputs.
# The naming convention is chronological: cleaned data -> SoW tables -> feature tables ->
# model outputs -> strategy outputs.
CLEANED_OUTPUT_PATH = BASE_DIR / "Transaction_Data_Cleaned.csv"
SOW_WINDOW_OUTPUT_PATH = BASE_DIR / "Transaction_Data_SoW_Eligible.csv"
LIFETIME_SOW_OUTPUT_PATH = BASE_DIR / "Customer_Lifetime_SoW.csv"
HALF_WINDOW_SOW_OUTPUT_PATH = BASE_DIR / "Customer_Half_Window_SoW.csv"
MONTHLY_SOW_OUTPUT_PATH = BASE_DIR / "Customer_Fiscal_Month_SoW.csv"
QUARTERLY_SOW_OUTPUT_PATH = BASE_DIR / "Customer_Fiscal_Quarter_SoW.csv"
CUSTOMER_FEATURES_OUTPUT_PATH = BASE_DIR / "Customer_Feature_Table.csv"
KMEANS_EVALUATION_OUTPUT_PATH = BASE_DIR / "KMeans_Model_Selection.csv"
SEGMENT_PROFILE_OUTPUT_PATH = BASE_DIR / "Customer_Segment_Profile.csv"
SEGMENTED_CUSTOMERS_OUTPUT_PATH = BASE_DIR / "Customer_Segment_and_Risk.csv"
SEGMENT_STRATEGY_OUTPUT_PATH = BASE_DIR / "Customer_Segment_Strategy.csv"
DECLINE_MODEL_METRICS_OUTPUT_PATH = BASE_DIR / "Decline_Model_Metrics.json"
LOGISTIC_COEFFICIENTS_OUTPUT_PATH = BASE_DIR / "Decline_Logistic_Coefficients.csv"
CALIBRATION_CURVE_OUTPUT_PATH = BASE_DIR / "Decline_Model_Calibration.csv"
SHAP_SUMMARY_OUTPUT_PATH = BASE_DIR / "Decline_Risk_SHAP_Summary.png"


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four raw CSVs used throughout the pipeline.

    Returning all four frames together keeps the orchestration in ``main()`` simple and makes
    it explicit that every later stage depends on the same shared raw inputs.
    """
    transactions = pd.read_csv(TRANSACTION_DATA_PATH)
    categories = pd.read_csv(CATEGORY_CODE_PATH)
    payment_codes = pd.read_csv(PAYMENT_CODE_PATH)
    customers = pd.read_csv(CUSTOMER_DATA_PATH)
    return transactions, categories, payment_codes, customers


def validate_primary_key(df: pd.DataFrame, column_name: str, dataset_name: str) -> None:
    """Fail fast if a supposed lookup-table primary key is broken.

    The lookup files are treated as many-to-one merge targets later in the pipeline. If one of
    these keys is null or duplicated, a merge could silently fan out rows or create missing
    mappings, so we enforce the contract up front.
    """
    # A null key would make the lookup fundamentally ambiguous.
    if df[column_name].isna().any():
        raise ValueError(f"{dataset_name}.{column_name} contains null values.")

    # Duplicate lookup keys would break the many-to-one merge assumptions downstream.
    if df[column_name].duplicated().any():
        duplicates = df.loc[df[column_name].duplicated(), column_name].tolist()
        raise ValueError(
            f"{dataset_name}.{column_name} is not unique. Duplicate values: {duplicates[:10]}"
        )


def parse_datetime_column(
    df: pd.DataFrame,
    column_name: str,
    date_format: str,
    allow_nulls: bool = False,
) -> None:
    """Parse a string date column in-place with strict format validation.

    ``errors='coerce'`` is used deliberately so invalid values become ``NaT`` and can then be
    checked explicitly. That gives us better control than letting pandas silently guess formats.
    """
    # We remember how many non-null source values existed so that optional-null columns can still
    # reject malformed strings while allowing genuine blanks.
    original_non_null_count = int(df[column_name].notna().sum())
    parsed = pd.to_datetime(df[column_name], format=date_format, errors="coerce")

    # Required date fields must parse cleanly for every row.
    if not allow_nulls and parsed.isna().any():
        raise ValueError(f"{column_name} contains invalid or missing dates.")

    # Optional date fields may be blank, but any non-blank value must still parse successfully.
    if allow_nulls and int(parsed.notna().sum()) != original_non_null_count:
        raise ValueError(f"{column_name} contains invalid dates.")

    df[column_name] = parsed


def validate_transaction_integrity(
    transactions: pd.DataFrame,
    categories: pd.DataFrame,
    payment_codes: pd.DataFrame,
) -> None:
    """Run the raw transaction sanity checks described in the case-study analysis.

    These checks encode facts we already validated manually:
    - transaction IDs are unique,
    - amounts are always positive before sign handling,
    - category / payment codes are fully covered by their lookup tables.

    Encoding them here prevents future data refreshes from silently violating those assumptions.
    """
    # Transaction_ID is expected to be unique at the rolled-up transaction-row grain.
    if transactions["Transaction_ID"].duplicated().any():
        duplicates = transactions.loc[
            transactions["Transaction_ID"].duplicated(), "Transaction_ID"
        ].tolist()
        raise ValueError(f"Duplicate Transaction_ID values found: {duplicates[:10]}")

    # Raw transaction amounts should all be positive; return direction is handled separately
    # through ``Transaction_Type`` when we create ``Net_Amount``.
    if not transactions["Transaction_Amount"].gt(0).all():
        raise ValueError("Transaction_Amount must be strictly positive for every row.")

    # If a transaction references an unknown category code, later category-mix features would be
    # incomplete or wrong.
    orphan_category_mask = ~transactions["Category_Code"].isin(categories["Category_Code"])
    if orphan_category_mask.any():
        orphan_codes = (
            transactions.loc[orphan_category_mask, "Category_Code"].drop_duplicates().tolist()
        )
        raise ValueError(f"Orphan Category_Code values found: {orphan_codes[:10]}")

    # Same logic for payment codes: the payment mix and ABC-card flags depend on this lookup.
    orphan_payment_mask = ~transactions["Payment_Code"].isin(payment_codes["Payment_Code"])
    if orphan_payment_mask.any():
        orphan_codes = (
            transactions.loc[orphan_payment_mask, "Payment_Code"].drop_duplicates().tolist()
        )
        raise ValueError(f"Orphan Payment_Code values found: {orphan_codes[:10]}")


def add_fiscal_fields(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Add fiscal month/year based on the case-study Aug-Jul fiscal calendar."""
    dated = df.copy()
    month_number = dated[date_column].dt.month
    # Fiscal month is remapped so Aug=1, Sep=2, ..., Jul=12.
    dated["Fiscal_Month"] = (((month_number - 8) % 12) + 1).astype("int64")
    # Fiscal year is labeled by the ending year of the Aug-Jul window.
    dated["Fiscal_Year"] = (
        dated[date_column].dt.year + (month_number >= 8).astype("int64")
    ).astype("int64")
    return dated


def add_fiscal_quarter(df: pd.DataFrame, fiscal_month_column: str = "Fiscal_Month") -> pd.DataFrame:
    """Derive fiscal quarter from an already-remapped fiscal month."""
    quartered = df.copy()
    quartered["Fiscal_Quarter"] = (
        ((quartered[fiscal_month_column] - 1) // 3) + 1
    ).astype("int64")
    return quartered


def compute_sow_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute a spend-share ratio with the project-wide denominator guard.

    A SoW-style ratio is only meaningful when total spend is strictly positive. We therefore
    convert any case with ``denominator <= 0`` to ``NaN`` rather than forcing a misleading
    negative / infinite / zero-filled value.
    """
    ratio = numerator.div(denominator)
    return ratio.mask(denominator.le(0))


def slugify_label(label: str) -> str:
    """Convert human-readable labels into safe snake_case column suffixes."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip().lower())
    return normalized.strip("_")


def merge_feature_block(
    features: pd.DataFrame,
    block: pd.DataFrame,
    block_name: str,
) -> pd.DataFrame:
    """Left-join a feature block and assert that customer grain is preserved.

    The most common failure mode in feature engineering is an accidental one-to-many merge that
    duplicates customers. This helper centralizes that guardrail.
    """
    merged = features.merge(
        block,
        on="Customer_ID",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(features):
        raise ValueError(f"Row count changed after merging {block_name}.")
    return merged


def summarize_spend(
    transactions: pd.DataFrame,
    group_keys: list[str],
    total_spend_column: str,
    abc_spend_column: str,
) -> pd.DataFrame:
    """Aggregate total and ABC-card spend at an arbitrary grain.

    This helper is reused for lifetime, half-window, monthly, and other rollups so that the
    spend aggregation logic stays consistent across the entire pipeline.
    """
    summarized = transactions.copy()
    # ``abc_net_amount`` isolates the numerator contribution without losing the full row context.
    summarized["abc_net_amount"] = summarized["Net_Amount"].where(
        summarized["is_abc_bank_credit_card"],
        0.0,
    )

    return (
        summarized.groupby(group_keys, as_index=False)
        .agg(
            **{
                total_spend_column: ("Net_Amount", "sum"),
                abc_spend_column: ("abc_net_amount", "sum"),
            }
        )
        .sort_values(group_keys, ignore_index=True)
    )


def clean_transaction_fields(transactions: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transaction rows and add analysis-ready transaction fields."""
    cleaned = transactions.copy()

    # Transaction dates in the raw file are stored as day-first strings like ``28-09-2025``.
    parse_datetime_column(cleaned, "Transaction_Date", "%d-%m-%Y")

    # Per the case rules, null ``Number_of_Transactions`` only occurs on returns and should be
    # mechanically filled with zero rather than statistically imputed.
    return_mask = cleaned["Transaction_Type"].eq("Return")
    null_count_mask = cleaned["Number_of_Transactions"].isna()
    invalid_null_mask = null_count_mask & ~return_mask
    if invalid_null_mask.any():
        raise ValueError("Only Return rows may have null Number_of_Transactions values.")

    # Once filled, the field can safely be treated as an integer transaction-count bundle size.
    cleaned.loc[return_mask, "Number_of_Transactions"] = 0
    cleaned["Number_of_Transactions"] = cleaned["Number_of_Transactions"].astype("int64")

    # The case study only expects these two transaction directions.
    valid_transaction_types = {"Sale", "Return"}
    invalid_types = sorted(set(cleaned["Transaction_Type"].dropna()) - valid_transaction_types)
    if invalid_types:
        raise ValueError(f"Unexpected Transaction_Type values found: {invalid_types}")

    # ``Net_Amount`` is the canonical spend measure used everywhere else in the analysis.
    # Sales stay positive; returns are made negative so later aggregations net automatically.
    cleaned["Net_Amount"] = cleaned["Transaction_Amount"].where(
        cleaned["Transaction_Type"].eq("Sale"),
        -cleaned["Transaction_Amount"],
    )

    return add_fiscal_fields(cleaned, "Transaction_Date")


def clean_customer_fields(customers: pd.DataFrame) -> pd.DataFrame:
    """Parse customer date fields and drop customers without an open date.

    Customers missing ``Credit_Card_Open_Date`` are excluded exactly as required by the data
    cleaning specification because we cannot define an active-card window for them.
    """
    cleaned = customers.copy()
    parse_datetime_column(
        cleaned,
        "Credit_Card_Open_Date",
        "%Y-%m-%d",
        allow_nulls=True,
    )
    parse_datetime_column(
        cleaned,
        "Credit_Card_Closed_Date",
        "%Y-%m-%d",
        allow_nulls=True,
    )

    return cleaned.dropna(subset=["Credit_Card_Open_Date"]).copy()


def merge_and_clean_transaction_data(
    transactions: pd.DataFrame,
    categories: pd.DataFrame,
    payment_codes: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, int, int]:
    """Run the full raw-data cleaning and enrichment stage.

    Returns both:
    - ``merged``: customer-joined transactions after cleaning, and
    - ``sow_eligible``: the subset that falls inside each customer's active-card window.
    """
    # Validate the reference tables before any merges happen.
    validate_primary_key(categories, "Category_Code", "Category Code")
    validate_primary_key(payment_codes, "Payment_Code", "Payment Code")
    validate_primary_key(customers, "Customer_ID", "Customer Data")
    validate_transaction_integrity(transactions, categories, payment_codes)

    # Standardize the transaction and customer tables independently first.
    cleaned_transactions = clean_transaction_fields(transactions)
    cleaned_customers = clean_customer_fields(customers)
    dropped_customer_count = len(customers) - len(cleaned_customers)

    # These are the three documented joins from the project brief.
    merged = cleaned_transactions.merge(
        categories,
        on="Category_Code",
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        payment_codes,
        on="Payment_Code",
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        cleaned_customers,
        on="Customer_ID",
        how="inner",
        validate="many_to_one",
    )

    # This count is a useful audit metric because it should match the transactions lost when the
    # no-open-date customers are excluded.
    dropped_transaction_count = len(cleaned_transactions) - len(merged)

    # This boolean is the SoW numerator flag used throughout the rest of the pipeline.
    merged["is_abc_bank_credit_card"] = merged["Payment_Method"].eq(
        "ABC Bank Credit Card"
    )

    # Study end is inferred directly from the observed transactions.
    study_end_date = merged["Transaction_Date"].max()

    # The active card end date is card close date when present, otherwise study end.
    merged["Active_Card_End_Date"] = merged["Credit_Card_Closed_Date"].fillna(study_end_date)
    merged["Active_Card_End_Date"] = merged["Active_Card_End_Date"].clip(upper=study_end_date)

    # This flag tells us whether a transaction is allowed to participate in SoW calculations.
    merged["is_in_active_card_period"] = merged["Transaction_Date"].between(
        merged["Credit_Card_Open_Date"],
        merged["Active_Card_End_Date"],
        inclusive="both",
    )

    # Keep the active-window subset separate because later stages need both:
    # - the full cleaned customer base, and
    # - the SoW-eligible transaction subset.
    sow_eligible = merged.loc[merged["is_in_active_card_period"]].copy()

    return (
        merged,
        sow_eligible,
        study_end_date,
        dropped_customer_count,
        dropped_transaction_count,
    )


def build_customer_observed_windows(cleaned_transactions: pd.DataFrame) -> pd.DataFrame:
    study_start_date = cleaned_transactions["Transaction_Date"].min()
    study_end_date = cleaned_transactions["Active_Card_End_Date"].max()

    customer_windows = cleaned_transactions[
        [
            "Customer_ID",
            "Credit_Card_Open_Date",
            "Credit_Card_Closed_Date",
            "Active_Card_End_Date",
        ]
    ].drop_duplicates().copy()

    customer_windows["Observed_Active_Start_Date"] = customer_windows[
        "Credit_Card_Open_Date"
    ].clip(lower=study_start_date)
    customer_windows["Observed_Active_End_Date"] = customer_windows[
        "Active_Card_End_Date"
    ].clip(upper=study_end_date)
    customer_windows["has_observed_active_window"] = (
        customer_windows["Observed_Active_Start_Date"]
        <= customer_windows["Observed_Active_End_Date"]
    )
    customer_windows["Observed_Active_Window_Days"] = (
        customer_windows["Observed_Active_End_Date"]
        - customer_windows["Observed_Active_Start_Date"]
    ).dt.days.add(1)
    customer_windows.loc[
        ~customer_windows["has_observed_active_window"],
        "Observed_Active_Window_Days",
    ] = 0

    return customer_windows.sort_values("Customer_ID", ignore_index=True)


def build_customer_month_panel(customer_windows: pd.DataFrame) -> pd.DataFrame:
    panel = customer_windows.loc[customer_windows["has_observed_active_window"]].copy()
    panel["Panel_Start_Date"] = panel["Observed_Active_Start_Date"]
    panel["Panel_End_Date"] = panel["Observed_Active_End_Date"]
    panel["Month_Start"] = panel["Panel_Start_Date"].dt.to_period("M").dt.to_timestamp()
    panel["Month_End"] = panel["Panel_End_Date"].dt.to_period("M").dt.to_timestamp()
    panel["Month_Start"] = panel.apply(
        lambda row: pd.date_range(row["Month_Start"], row["Month_End"], freq="MS"),
        axis=1,
    )
    panel = panel.explode("Month_Start", ignore_index=True)
    panel["Month_End"] = panel["Month_Start"] + MonthEnd(1)
    panel = add_fiscal_fields(panel, "Month_Start")
    return add_fiscal_quarter(panel)


def compute_lifetime_share_of_wallet(
    cleaned_transactions: pd.DataFrame,
    sow_eligible_transactions: pd.DataFrame,
) -> pd.DataFrame:
    customer_windows = build_customer_observed_windows(cleaned_transactions)
    lifetime_spend = summarize_spend(
        transactions=sow_eligible_transactions,
        group_keys=["Customer_ID"],
        total_spend_column="lifetime_total_spend",
        abc_spend_column="lifetime_abc_spend",
    )

    lifetime_sow = customer_windows.merge(
        lifetime_spend,
        on="Customer_ID",
        how="left",
        validate="one_to_one",
    )
    lifetime_sow["lifetime_total_spend"] = lifetime_sow["lifetime_total_spend"].fillna(0.0)
    lifetime_sow["lifetime_abc_spend"] = lifetime_sow["lifetime_abc_spend"].fillna(0.0)
    lifetime_sow["SoW_lifetime"] = compute_sow_ratio(
        lifetime_sow["lifetime_abc_spend"],
        lifetime_sow["lifetime_total_spend"],
    )
    lifetime_sow["is_abc_return_dominated"] = lifetime_sow["lifetime_abc_spend"].lt(0)
    lifetime_sow["is_cross_channel_return_skewed"] = lifetime_sow[
        "lifetime_abc_spend"
    ].gt(lifetime_sow["lifetime_total_spend"]) & lifetime_sow[
        "lifetime_total_spend"
    ].gt(0)
    lifetime_sow["use_for_bounded_features"] = (
        lifetime_sow["lifetime_total_spend"].gt(0)
        & ~lifetime_sow["is_abc_return_dominated"]
        & ~lifetime_sow["is_cross_channel_return_skewed"]
    )
    lifetime_sow["sow_out_of_bounds"] = (
        lifetime_sow["SoW_lifetime"].gt(1) | lifetime_sow["SoW_lifetime"].lt(0)
    )

    return lifetime_sow.sort_values("Customer_ID", ignore_index=True)


def compute_half_window_share_of_wallet(
    cleaned_transactions: pd.DataFrame,
    sow_eligible_transactions: pd.DataFrame,
) -> pd.DataFrame:
    customer_windows = build_customer_observed_windows(cleaned_transactions)
    customer_windows["Active_Window_Midpoint"] = customer_windows[
        "Observed_Active_Start_Date"
    ] + (
        customer_windows["Observed_Active_End_Date"]
        - customer_windows["Observed_Active_Start_Date"]
    ) / 2
    customer_windows.loc[
        ~customer_windows["has_observed_active_window"],
        "Active_Window_Midpoint",
    ] = pd.NaT

    half_base = customer_windows.assign(_merge_key=1).merge(
        pd.DataFrame({"Window_Half": ["H1", "H2"], "_merge_key": [1, 1]}),
        on="_merge_key",
        how="inner",
        validate="many_to_many",
    )
    half_base = half_base.drop(columns="_merge_key")

    half_spend = sow_eligible_transactions.merge(
        customer_windows[["Customer_ID", "Active_Window_Midpoint"]],
        on="Customer_ID",
        how="left",
        validate="many_to_one",
    )
    half_spend["Window_Half"] = half_spend["Transaction_Date"].le(
        half_spend["Active_Window_Midpoint"]
    ).map({True: "H1", False: "H2"})

    half_spend = summarize_spend(
        transactions=half_spend,
        group_keys=["Customer_ID", "Window_Half"],
        total_spend_column="half_total_spend",
        abc_spend_column="half_abc_spend",
    )

    half_panel = half_base.merge(
        half_spend,
        on=["Customer_ID", "Window_Half"],
        how="left",
        validate="one_to_one",
    )
    half_panel["half_total_spend"] = half_panel["half_total_spend"].fillna(0.0)
    half_panel["half_abc_spend"] = half_panel["half_abc_spend"].fillna(0.0)
    half_panel["SoW_half"] = compute_sow_ratio(
        half_panel["half_abc_spend"],
        half_panel["half_total_spend"],
    )
    half_panel["is_abc_return_dominated"] = half_panel["half_abc_spend"].lt(0)
    half_panel["is_cross_channel_return_skewed"] = half_panel["half_abc_spend"].gt(
        half_panel["half_total_spend"]
    ) & half_panel["half_total_spend"].gt(0)
    half_panel["use_for_bounded_features"] = (
        half_panel["half_total_spend"].gt(0)
        & ~half_panel["is_abc_return_dominated"]
        & ~half_panel["is_cross_channel_return_skewed"]
    )
    half_panel["sow_out_of_bounds"] = (
        half_panel["SoW_half"].gt(1) | half_panel["SoW_half"].lt(0)
    )

    first_half = half_panel.loc[half_panel["Window_Half"].eq("H1")].drop(
        columns="Window_Half"
    )
    first_half = first_half.rename(
        columns={
            "half_total_spend": "H1_total_spend",
            "half_abc_spend": "H1_abc_spend",
            "SoW_half": "SoW_H1",
            "is_abc_return_dominated": "H1_is_abc_return_dominated",
            "is_cross_channel_return_skewed": "H1_is_cross_channel_return_skewed",
            "use_for_bounded_features": "H1_use_for_bounded_features",
            "sow_out_of_bounds": "H1_sow_out_of_bounds",
        }
    )

    second_half = half_panel.loc[half_panel["Window_Half"].eq("H2")].drop(
        columns="Window_Half"
    )
    second_half = second_half.rename(
        columns={
            "half_total_spend": "H2_total_spend",
            "half_abc_spend": "H2_abc_spend",
            "SoW_half": "SoW_H2",
            "is_abc_return_dominated": "H2_is_abc_return_dominated",
            "is_cross_channel_return_skewed": "H2_is_cross_channel_return_skewed",
            "use_for_bounded_features": "H2_use_for_bounded_features",
            "sow_out_of_bounds": "H2_sow_out_of_bounds",
        }
    )

    half_window_sow = first_half.merge(
        second_half[
            [
                "Customer_ID",
                "H2_total_spend",
                "H2_abc_spend",
                "SoW_H2",
                "H2_is_abc_return_dominated",
                "H2_is_cross_channel_return_skewed",
                "H2_use_for_bounded_features",
                "H2_sow_out_of_bounds",
            ]
        ],
        on="Customer_ID",
        how="left",
        validate="one_to_one",
    )
    half_window_sow["SoW_H2_minus_H1"] = (
        half_window_sow["SoW_H2"] - half_window_sow["SoW_H1"]
    )
    half_window_sow["use_half_window_delta_for_bounded_features"] = (
        half_window_sow["H1_use_for_bounded_features"]
        & half_window_sow["H2_use_for_bounded_features"]
        & half_window_sow["SoW_H1"].notna()
        & half_window_sow["SoW_H2"].notna()
    )

    return half_window_sow.sort_values("Customer_ID", ignore_index=True)


def compute_monthly_share_of_wallet(
    cleaned_transactions: pd.DataFrame,
    sow_eligible_transactions: pd.DataFrame,
) -> pd.DataFrame:
    customer_windows = build_customer_observed_windows(cleaned_transactions)
    customer_month_panel = build_customer_month_panel(customer_windows)

    transaction_monthly = sow_eligible_transactions.copy()
    transaction_monthly["Month_Start"] = (
        transaction_monthly["Transaction_Date"].dt.to_period("M").dt.to_timestamp()
    )

    monthly_spend = summarize_spend(
        transactions=transaction_monthly,
        group_keys=["Customer_ID", "Month_Start", "Fiscal_Year", "Fiscal_Month"],
        total_spend_column="monthly_total_spend",
        abc_spend_column="monthly_abc_spend",
    )

    monthly_sow = customer_month_panel.merge(
        monthly_spend,
        on=["Customer_ID", "Month_Start", "Fiscal_Year", "Fiscal_Month"],
        how="left",
        validate="one_to_one",
    )
    monthly_sow["monthly_total_spend"] = monthly_sow["monthly_total_spend"].fillna(0.0)
    monthly_sow["monthly_abc_spend"] = monthly_sow["monthly_abc_spend"].fillna(0.0)
    monthly_sow["SoW_month"] = compute_sow_ratio(
        monthly_sow["monthly_abc_spend"],
        monthly_sow["monthly_total_spend"],
    )
    monthly_sow["sow_out_of_bounds"] = (
        monthly_sow["SoW_month"].gt(1) | monthly_sow["SoW_month"].lt(0)
    )

    return monthly_sow.sort_values(["Customer_ID", "Month_Start"], ignore_index=True)


def compute_quarterly_share_of_wallet(monthly_sow: pd.DataFrame) -> pd.DataFrame:
    panel = add_fiscal_quarter(monthly_sow)
    quarterly_sow = (
        panel.groupby(
            ["Customer_ID", "Fiscal_Year", "Fiscal_Quarter"],
            as_index=False,
        )
        .agg(
            quarterly_total_spend=("monthly_total_spend", "sum"),
            quarterly_abc_spend=("monthly_abc_spend", "sum"),
        )
        .sort_values(["Customer_ID", "Fiscal_Year", "Fiscal_Quarter"], ignore_index=True)
    )

    quarterly_sow["SoW_quarter"] = compute_sow_ratio(
        quarterly_sow["quarterly_abc_spend"],
        quarterly_sow["quarterly_total_spend"],
    )
    quarterly_sow["sow_out_of_bounds"] = (
        quarterly_sow["SoW_quarter"].gt(1) | quarterly_sow["SoW_quarter"].lt(0)
    )

    return quarterly_sow


def compute_customer_feature_table(
    customers: pd.DataFrame,
    study_end_date: pd.Timestamp,
    sow_eligible_transactions: pd.DataFrame,
    lifetime_sow: pd.DataFrame,
    half_window_sow: pd.DataFrame,
) -> pd.DataFrame:
    cleaned_customers = clean_customer_fields(customers)
    features = cleaned_customers[
        [
            "Customer_ID",
            "Age",
            "Gender",
            "Membership_Type",
            "Credit_Card_Open_Date",
            "Credit_Card_Closed_Date",
            "Credit_Card_Limit",
            "Credit_Card_APR",
        ]
    ].copy()
    features["is_churned"] = features["Credit_Card_Closed_Date"].notna()
    features["Active_Card_End_Date"] = features["Credit_Card_Closed_Date"].fillna(study_end_date)
    features["Active_Card_End_Date"] = features["Active_Card_End_Date"].clip(upper=study_end_date)
    features["tenure_days"] = (
        features["Active_Card_End_Date"] - features["Credit_Card_Open_Date"]
    ).dt.days.astype("int64")

    last_transaction = (
        sow_eligible_transactions.groupby("Customer_ID", as_index=False)["Transaction_Date"]
        .max()
        .rename(columns={"Transaction_Date": "last_transaction_date"})
    )
    features = merge_feature_block(features, last_transaction, "last_transaction")
    features["recency_days"] = (
        study_end_date - features["last_transaction_date"]
    ).dt.days
    features["recency_days"] = features["recency_days"].fillna(features["tenure_days"])
    features["recency_days"] = features["recency_days"].astype("int64")

    rfm = (
        sow_eligible_transactions.groupby("Customer_ID", as_index=False)
        .agg(
            frequency=("Number_of_Transactions", "sum"),
            monetary=("Net_Amount", "sum"),
            avg_ticket=("Net_Amount", "mean"),
            n_active_rows=("Transaction_ID", "count"),
        )
        .sort_values("Customer_ID", ignore_index=True)
    )
    features = merge_feature_block(features, rfm, "rfm")
    features["frequency"] = features["frequency"].fillna(0).astype("int64")
    features["monetary"] = features["monetary"].fillna(0.0)
    features["avg_ticket"] = features["avg_ticket"].fillna(0.0)
    features["n_active_rows"] = features["n_active_rows"].fillna(0).astype("int64")

    monetary_check = lifetime_sow[["Customer_ID", "lifetime_total_spend"]].copy()
    features = merge_feature_block(features, monetary_check, "lifetime_total_spend_check")
    if not features["monetary"].equals(features["lifetime_total_spend"]):
        raise ValueError("RFM monetary does not match lifetime_total_spend.")

    category_mix = sow_eligible_transactions.copy()
    category_mix["Category_Group"] = category_mix["Category"].where(
        category_mix["Category"].isin(["Grocery", "Electronics"]),
        "Other",
    )
    category_spend = (
        category_mix.pivot_table(
            index="Customer_ID",
            columns="Category_Group",
            values="Net_Amount",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=["Grocery", "Electronics", "Other"], fill_value=0.0)
        .reset_index()
    )
    category_spend["grocery_pct"] = compute_sow_ratio(
        category_spend["Grocery"],
        category_spend["Customer_ID"].map(features.set_index("Customer_ID")["monetary"]),
    )
    category_spend["electronics_pct"] = compute_sow_ratio(
        category_spend["Electronics"],
        category_spend["Customer_ID"].map(features.set_index("Customer_ID")["monetary"]),
    )
    category_spend["other_pct"] = compute_sow_ratio(
        category_spend["Other"],
        category_spend["Customer_ID"].map(features.set_index("Customer_ID")["monetary"]),
    )
    category_pct = category_spend[
        ["Customer_ID", "grocery_pct", "electronics_pct", "other_pct"]
    ].copy()
    features = merge_feature_block(features, category_pct, "category_pct")

    payment_spend = (
        sow_eligible_transactions.pivot_table(
            index="Customer_ID",
            columns="Payment_Method",
            values="Net_Amount",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    payment_pct_columns: list[str] = []
    monetary_by_customer = features.set_index("Customer_ID")["monetary"]
    for payment_method in payment_spend.columns:
        if payment_method == "Customer_ID":
            continue
        payment_pct_column = f"payment_pct_{slugify_label(payment_method)}"
        payment_spend[payment_pct_column] = compute_sow_ratio(
            payment_spend[payment_method],
            payment_spend["Customer_ID"].map(monetary_by_customer),
        )
        payment_pct_columns.append(payment_pct_column)
    payment_pct = payment_spend[["Customer_ID", *payment_pct_columns]].copy()
    features = merge_feature_block(features, payment_pct, "payment_pct")

    sow_features = lifetime_sow[
        [
            "Customer_ID",
            "SoW_lifetime",
            "use_for_bounded_features",
            "is_abc_return_dominated",
            "is_cross_channel_return_skewed",
            "sow_out_of_bounds",
        ]
    ].rename(columns={"sow_out_of_bounds": "lifetime_sow_out_of_bounds"})
    features = merge_feature_block(features, sow_features, "lifetime_sow_features")

    half_window_features = half_window_sow[
        [
            "Customer_ID",
            "SoW_H2_minus_H1",
            "use_half_window_delta_for_bounded_features",
        ]
    ].copy()
    features = merge_feature_block(features, half_window_features, "half_window_features")

    features["trend_status"] = "Undetermined"
    declining_mask = (
        features["use_half_window_delta_for_bounded_features"]
        & features["SoW_H2_minus_H1"].le(-0.05)
    )
    improving_mask = (
        features["use_half_window_delta_for_bounded_features"]
        & features["SoW_H2_minus_H1"].ge(0.05)
    )
    stable_mask = (
        features["use_half_window_delta_for_bounded_features"]
        & ~declining_mask
        & ~improving_mask
    )
    features.loc[declining_mask, "trend_status"] = "Declining"
    features.loc[improving_mask, "trend_status"] = "Improving"
    features.loc[stable_mask, "trend_status"] = "Stable"

    features["has_measurable_sow"] = features["SoW_lifetime"].notna()
    features["is_dormant_cardholder"] = features["n_active_rows"].eq(0)
    features = features.drop(columns=["lifetime_total_spend"])

    if len(features) != 38164:
        raise ValueError(f"Customer feature table row count is {len(features)}, expected 38164.")

    return features.sort_values("Customer_ID", ignore_index=True)


def top_decile_precision(y_true: pd.Series, scores: np.ndarray) -> float:
    if len(scores) == 0:
        return float("nan")
    top_count = max(1, int(np.ceil(len(scores) * 0.10)))
    top_indices = np.argsort(scores)[-top_count:]
    return float(y_true.iloc[top_indices].mean())


def prettify_shap_feature_name(raw_feature_name: str) -> str:
    if raw_feature_name.startswith("num__"):
        return raw_feature_name.replace("num__", "", 1)

    if raw_feature_name.startswith("cat__"):
        readable = raw_feature_name.replace("cat__", "", 1)
        for prefix in ("Gender_", "Membership_Type_"):
            if readable.startswith(prefix):
                field_name = prefix[:-1]
                field_value = readable.replace(prefix, "", 1)
                return f"{field_name}={field_value}"
        return readable

    return raw_feature_name


def choose_cluster_name(
    profile: pd.Series,
    spend_thresholds: tuple[float, float],
    sow_thresholds: tuple[float, float],
) -> str:
    monetary_low, monetary_high = spend_thresholds
    sow_low, sow_high = sow_thresholds

    if profile["monetary"] >= monetary_high:
        spend_label = "High-Value"
    elif profile["monetary"] <= monetary_low:
        spend_label = "Value"
    else:
        spend_label = "Core"

    category_shares = {
        "Grocery": profile["grocery_pct"],
        "Electronics": profile["electronics_pct"],
        "Mixed": profile["other_pct"],
    }
    category_label = max(category_shares, key=category_shares.get)
    if category_label == "Mixed":
        category_label = "Mixed-Basket"

    payment_labels = {
        "payment_pct_abc_bank_credit_card": "ABC Loyalists",
        "payment_pct_cash_upi": "Cash/UPI Users",
        "payment_pct_debit_card": "Debit-First Users",
        "payment_pct_other_bank_credit_card": "Cross-Card Users",
    }
    top_payment_feature = max(payment_labels, key=lambda column: profile[column])
    payment_label = payment_labels[top_payment_feature]

    if profile["SoW_lifetime"] >= sow_high and payment_label == "ABC Loyalists":
        name = f"{spend_label} {category_label} ABC Loyalists"
    elif profile["SoW_lifetime"] <= sow_low and payment_label == "Cross-Card Users":
        name = f"{spend_label} {category_label} Cross-Card Drifters"
    elif profile["SoW_lifetime"] <= sow_low and payment_label == "Cash/UPI Users":
        name = f"{spend_label} {category_label} Cash Drifters"
    else:
        name = f"{spend_label} {category_label} {payment_label}"

    return name


def compute_segment_profiles(segmented_customers: pd.DataFrame) -> pd.DataFrame:
    profile_rows: list[dict[str, float | int | str]] = []

    for segment_name, group in segmented_customers.groupby("cluster_segment", sort=False):
        bounded_sow = group.loc[group["use_for_bounded_features"], "SoW_lifetime"]
        flagged_count = int(
            (group["has_measurable_sow"] & ~group["use_for_bounded_features"]).sum()
        )
        profile_rows.append(
            {
                "cluster_segment": segment_name,
                "customer_count": int(group["Customer_ID"].count()),
                "measurable_sow_count": int(group["has_measurable_sow"].sum()),
                "bounded_sow_count": int(group["use_for_bounded_features"].sum()),
                "flagged_count": flagged_count,
                "flagged_share": float(flagged_count / len(group)),
                "avg_age": float(group["Age"].mean()),
                "avg_credit_limit": float(group["Credit_Card_Limit"].mean()),
                "avg_apr": float(group["Credit_Card_APR"].mean()),
                "avg_tenure_days": float(group["tenure_days"].mean()),
                "avg_recency_days": float(group["recency_days"].mean()),
                "avg_frequency": float(group["frequency"].mean()),
                "avg_monetary": float(group["monetary"].mean()),
                "avg_ticket": float(group["avg_ticket"].mean()),
                "avg_sow_lifetime": float(bounded_sow.mean()) if not bounded_sow.empty else np.nan,
                "declining_share": float((group["trend_status"] == "Declining").mean()),
                "dormant_share": float(group["is_dormant_cardholder"].mean()),
                "prime_share": float((group["Membership_Type"] == "Prime").mean()),
            }
        )

    profile = pd.DataFrame(profile_rows).sort_values(
        "customer_count",
        ascending=False,
        ignore_index=True,
    )
    return profile


def compute_segment_strategy(segment_profiles: pd.DataFrame) -> pd.DataFrame:
    strategy_map = {
        "High-Value Mixed-Basket ABC Loyalists": {
            "segment_story": (
                "Biggest revenue base and the single largest raw pool of declining customers."
            ),
            "recommended_offer": (
                "Risk-score-targeted bonus cashback on Grocery/Electronics plus a Prime-upgrade nudge."
            ),
            "strategy_objective": "Retention",
        },
        "Core Mixed-Basket Cash/UPI Users": {
            "segment_story": (
                "Already heavily drifted to cash/UPI, so this is a reactivation and win-back segment."
            ),
            "recommended_offer": (
                "Win-back bonus with point-of-sale reminders about no annual fee and category-leading rewards."
            ),
            "strategy_objective": "Win-Back",
        },
        "Core Mixed-Basket ABC Loyalists": {
            "segment_story": (
                "Highest corrected SoW segment with meaningful decline risk that should be defended efficiently."
            ),
            "recommended_offer": (
                "Loyalty and milestone rewards rather than broad discounts to protect margin."
            ),
            "strategy_objective": "Loyalty Defense",
        },
        "High-Value Electronics Cross-Card Users": {
            "segment_story": (
                "High spenders are routing valuable electronics purchases to competing cards."
            ),
            "recommended_offer": (
                "Time-boxed electronics cashback bonus to recapture competitor-card electronics spend."
            ),
            "strategy_objective": "Cross-Card Capture",
        },
        "Value Mixed-Basket Debit-First Users": {
            "segment_story": (
                "Debit-card habit is the main barrier, not a lack of relevant category spend."
            ),
            "recommended_offer": (
                "In-app default-payment nudge paired with switching-friction cashback."
            ),
            "strategy_objective": "Behavior Shift",
        },
        "Dormant Cardholders": {
            "segment_story": "Card opened but never activated, so the priority is first-use activation.",
            "recommended_offer": "First-transaction welcome bonus.",
            "strategy_objective": "Activation",
        },
        "Return-Heavy/Anomalous": {
            "segment_story": (
                "Returns outweigh spend, so this is better handled operationally than with a marketing incentive."
            ),
            "recommended_offer": "Route to service or fraud review instead of a standard campaign.",
            "strategy_objective": "Service Review",
        },
        "Value Grocery Cash Drifters": {
            "segment_story": (
                "Grocery-heavy and cash-preferring, which makes this one of the cheapest reward-rate wins."
            ),
            "recommended_offer": (
                "Grocery threshold bonus that nudges cash transactions onto the ABC card."
            ),
            "strategy_objective": "Category Win-Back",
        },
    }

    strategy = segment_profiles.copy()
    strategy["segment_story"] = strategy["cluster_segment"].map(
        lambda segment: strategy_map.get(segment, {}).get(
            "segment_story",
            "Review the segment profile manually before assigning a campaign.",
        )
    )
    strategy["recommended_offer"] = strategy["cluster_segment"].map(
        lambda segment: strategy_map.get(segment, {}).get(
            "recommended_offer",
            "Manual review required.",
        )
    )
    strategy["strategy_objective"] = strategy["cluster_segment"].map(
        lambda segment: strategy_map.get(segment, {}).get(
            "strategy_objective",
            "Review",
        )
    )
    return strategy


def fit_customer_segments(
    customer_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segmentation = customer_features[
        [
            "Customer_ID",
            "is_dormant_cardholder",
            "has_measurable_sow",
            "is_abc_return_dominated",
            "is_cross_channel_return_skewed",
            "trend_status",
            "Age",
            "Membership_Type",
            "Credit_Card_Limit",
            "Credit_Card_APR",
            "tenure_days",
            "recency_days",
            "frequency",
            "monetary",
            "avg_ticket",
            "grocery_pct",
            "electronics_pct",
            "other_pct",
            "payment_pct_abc_bank_credit_card",
            "payment_pct_cash_upi",
            "payment_pct_debit_card",
            "payment_pct_other_bank_credit_card",
            "payment_pct_xyz_wallet",
            "SoW_lifetime",
        ]
    ].copy()

    segmentation["segment_population"] = np.select(
        [
            segmentation["is_dormant_cardholder"],
            ~segmentation["has_measurable_sow"],
        ],
        [
            "Dormant",
            "Return-Heavy/Anomalous",
        ],
        default="Measurable-SoW",
    )
    segmentation["cluster_segment"] = np.select(
        [
            segmentation["is_dormant_cardholder"],
            ~segmentation["has_measurable_sow"],
        ],
        [
            "Dormant Cardholders",
            "Return-Heavy/Anomalous",
        ],
        default=pd.NA,
    )

    cluster_feature_columns = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_ticket",
        "tenure_days",
        "grocery_pct",
        "electronics_pct",
        "other_pct",
        "payment_pct_abc_bank_credit_card",
        "payment_pct_cash_upi",
        "payment_pct_debit_card",
        "payment_pct_other_bank_credit_card",
        "SoW_lifetime",
    ]
    log_columns = ["recency_days", "frequency", "monetary", "avg_ticket"]

    measurable_population = segmentation.loc[
        segmentation["segment_population"].eq("Measurable-SoW")
    ].copy()
    clustering_frame = measurable_population[cluster_feature_columns].astype("float64").copy()
    clustering_frame.loc[:, log_columns] = np.log1p(clustering_frame[log_columns])
    winsor_lower = clustering_frame.quantile(0.005)
    winsor_upper = clustering_frame.quantile(0.995)
    clustering_frame = clustering_frame.clip(
        lower=winsor_lower,
        upper=winsor_upper,
        axis=1,
    )

    scaler = StandardScaler()
    clustering_matrix = scaler.fit_transform(clustering_frame)

    k_evaluation_rows: list[dict[str, float | int]] = []
    fitted_models: dict[int, tuple[KMeans, np.ndarray]] = {}
    for k in range(4, 7):
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = kmeans.fit_predict(clustering_matrix)
        silhouette = silhouette_score(
            clustering_matrix,
            labels,
            sample_size=min(10000, len(clustering_matrix)),
            random_state=42,
        )
        fitted_models[k] = (kmeans, labels)
        k_evaluation_rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette_score": float(silhouette),
            }
        )

    kmeans_evaluation = pd.DataFrame(k_evaluation_rows).sort_values("k", ignore_index=True)
    best_k_row = kmeans_evaluation.sort_values(
        ["silhouette_score", "k"],
        ascending=[False, True],
        ignore_index=True,
    ).iloc[0]
    best_k = int(best_k_row["k"])
    _, best_labels = fitted_models[best_k]
    measurable_population["cluster_id"] = best_labels

    cluster_profiles = (
        measurable_population.groupby("cluster_id", as_index=False)[cluster_feature_columns]
        .mean()
        .sort_values("cluster_id", ignore_index=True)
    )
    spend_thresholds = (
        float(cluster_profiles["monetary"].quantile(0.33)),
        float(cluster_profiles["monetary"].quantile(0.67)),
    )
    sow_thresholds = (
        float(cluster_profiles["SoW_lifetime"].quantile(0.33)),
        float(cluster_profiles["SoW_lifetime"].quantile(0.67)),
    )

    cluster_name_map: dict[int, str] = {}
    used_names: set[str] = set()
    for _, profile in cluster_profiles.iterrows():
        base_name = choose_cluster_name(profile, spend_thresholds, sow_thresholds)
        candidate_name = base_name
        suffix = 2
        while candidate_name in used_names:
            candidate_name = f"{base_name} {suffix}"
            suffix += 1
        cluster_name_map[int(profile["cluster_id"])] = candidate_name
        used_names.add(candidate_name)

    measurable_population["cluster_segment"] = measurable_population["cluster_id"].map(
        cluster_name_map
    )
    segmentation["cluster_id"] = pd.NA
    measurable_cluster_map = measurable_population.set_index("Customer_ID")
    measurable_mask = segmentation["segment_population"].eq("Measurable-SoW")
    segmentation.loc[measurable_mask, "cluster_id"] = segmentation.loc[
        measurable_mask, "Customer_ID"
    ].map(measurable_cluster_map["cluster_id"])
    segmentation.loc[measurable_mask, "cluster_segment"] = segmentation.loc[
        measurable_mask, "Customer_ID"
    ].map(measurable_cluster_map["cluster_segment"])

    profile_source = customer_features.merge(
        segmentation[["Customer_ID", "segment_population", "cluster_segment"]],
        on="Customer_ID",
        how="left",
        validate="one_to_one",
    )
    segment_profiles = compute_segment_profiles(profile_source)

    kmeans_evaluation["selected_k"] = kmeans_evaluation["k"].eq(best_k)

    return (
        segmentation[
            ["Customer_ID", "segment_population", "cluster_id", "cluster_segment"]
        ].sort_values("Customer_ID", ignore_index=True),
        kmeans_evaluation,
        segment_profiles,
    )


def train_decline_classifier(
    customer_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    classifier_population = customer_features.loc[
        customer_features["trend_status"].isin(["Declining", "Stable", "Improving"])
    ].copy()
    classifier_population["is_declining"] = classifier_population["trend_status"].eq(
        "Declining"
    ).astype("int64")

    numeric_features = [
        "Age",
        "Credit_Card_Limit",
        "Credit_Card_APR",
        "tenure_days",
        "recency_days",
        "frequency",
        "monetary",
        "avg_ticket",
        "grocery_pct",
        "electronics_pct",
        "other_pct",
        "payment_pct_abc_bank_credit_card",
        "payment_pct_cash_upi",
        "payment_pct_debit_card",
        "payment_pct_other_bank_credit_card",
        "payment_pct_xyz_wallet",
        "SoW_lifetime",
    ]
    categorical_features = ["Gender", "Membership_Type"]
    feature_columns = numeric_features + categorical_features

    X = classifier_population[feature_columns].copy()
    y = classifier_population["is_declining"].copy()

    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        X,
        y,
        classifier_population["Customer_ID"],
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    logistic_model.fit(X_train, y_train)
    logistic_scores = logistic_model.predict_proba(X_test)[:, 1]

    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)
    transformed_feature_names = [
        prettify_shap_feature_name(name)
        for name in preprocessor.get_feature_names_out()
    ]

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
    )
    xgb_model.fit(transformed_train, y_train)
    xgb_scores = xgb_model.predict_proba(transformed_test)[:, 1]
    xgb_predictions = (xgb_scores >= 0.50).astype("int64")

    calibration_expected, calibration_predicted = calibration_curve(
        y_test,
        xgb_scores,
        n_bins=10,
        strategy="quantile",
    )
    calibration_frame = pd.DataFrame(
        {
            "mean_predicted_probability": calibration_predicted,
            "observed_decline_rate": calibration_expected,
        }
    )

    logistic_coefficients = pd.DataFrame(
        {
            "feature_name": transformed_feature_names,
            "coefficient": logistic_model.named_steps["model"].coef_[0],
        }
    ).sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "declining_rows": int(y.sum()),
        "non_declining_rows": int((1 - y).sum()),
        "logistic_auc_roc": float(roc_auc_score(y_test, logistic_scores)),
        "logistic_average_precision": float(average_precision_score(y_test, logistic_scores)),
        "logistic_top_decile_precision": float(top_decile_precision(y_test.reset_index(drop=True), logistic_scores)),
        "xgb_auc_roc": float(roc_auc_score(y_test, xgb_scores)),
        "xgb_average_precision": float(average_precision_score(y_test, xgb_scores)),
        "xgb_precision_at_0_5": float(precision_score(y_test, xgb_predictions, zero_division=0)),
        "xgb_recall_at_0_5": float(recall_score(y_test, xgb_predictions, zero_division=0)),
        "xgb_top_decile_precision": float(top_decile_precision(y_test.reset_index(drop=True), xgb_scores)),
        "xgb_brier_score": float(brier_score_loss(y_test, xgb_scores)),
    }

    full_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    X_full = classifier_population[feature_columns].copy()
    transformed_full = full_preprocessor.fit_transform(X_full)
    full_feature_names = [
        prettify_shap_feature_name(name)
        for name in full_preprocessor.get_feature_names_out()
    ]

    final_xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
    )
    final_xgb_model.fit(transformed_full, y)
    decline_scores = final_xgb_model.predict_proba(transformed_full)[:, 1]

    full_dmatrix = xgb.DMatrix(transformed_full, feature_names=full_feature_names)
    shap_contributions = final_xgb_model.get_booster().predict(
        full_dmatrix,
        pred_contribs=True,
    )
    shap_matrix = shap_contributions[:, :-1]

    top_driver_names: list[list[str]] = []
    top_driver_impacts: list[list[float]] = []
    for row_index in range(shap_matrix.shape[0]):
        row_values = shap_matrix[row_index]
        top_indices = np.argsort(np.abs(row_values))[-3:][::-1]
        top_driver_names.append([full_feature_names[index] for index in top_indices])
        top_driver_impacts.append([float(row_values[index]) for index in top_indices])

    shap_output = classifier_population[
        ["Customer_ID", "trend_status", "is_declining"]
    ].copy()
    shap_output["decline_risk_score"] = decline_scores
    shap_output["top_shap_driver_1"] = [drivers[0] for drivers in top_driver_names]
    shap_output["top_shap_driver_2"] = [drivers[1] for drivers in top_driver_names]
    shap_output["top_shap_driver_3"] = [drivers[2] for drivers in top_driver_names]
    shap_output["top_shap_impact_1"] = [impacts[0] for impacts in top_driver_impacts]
    shap_output["top_shap_impact_2"] = [impacts[1] for impacts in top_driver_impacts]
    shap_output["top_shap_impact_3"] = [impacts[2] for impacts in top_driver_impacts]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_matrix,
        transformed_full,
        feature_names=full_feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    test_scores = pd.DataFrame(
        {
            "Customer_ID": test_ids.values,
            "actual_is_declining": y_test.values,
            "logistic_score": logistic_scores,
            "xgb_score": xgb_scores,
        }
    )
    metrics["test_top_decile_customers"] = int(max(1, np.ceil(len(test_scores) * 0.10)))

    return shap_output.sort_values("Customer_ID", ignore_index=True), metrics, logistic_coefficients, calibration_frame


def main() -> None:
    transactions, categories, payment_codes, customers = load_dataframes()
    (
        cleaned_transactions,
        sow_eligible_transactions,
        study_end_date,
        dropped_customer_count,
        dropped_transaction_count,
    ) = merge_and_clean_transaction_data(
        transactions=transactions,
        categories=categories,
        payment_codes=payment_codes,
        customers=customers,
    )

    lifetime_sow = compute_lifetime_share_of_wallet(
        cleaned_transactions=cleaned_transactions,
        sow_eligible_transactions=sow_eligible_transactions,
    )
    half_window_sow = compute_half_window_share_of_wallet(
        cleaned_transactions=cleaned_transactions,
        sow_eligible_transactions=sow_eligible_transactions,
    )
    monthly_sow = compute_monthly_share_of_wallet(
        cleaned_transactions=cleaned_transactions,
        sow_eligible_transactions=sow_eligible_transactions,
    )
    quarterly_sow = compute_quarterly_share_of_wallet(monthly_sow)
    customer_features = compute_customer_feature_table(
        customers=customers,
        study_end_date=study_end_date,
        sow_eligible_transactions=sow_eligible_transactions,
        lifetime_sow=lifetime_sow,
        half_window_sow=half_window_sow,
    )
    customer_segments, kmeans_evaluation, segment_profiles = fit_customer_segments(
        customer_features
    )
    segment_strategy = compute_segment_strategy(segment_profiles)
    decline_scoring, decline_metrics, logistic_coefficients, calibration_curve_data = (
        train_decline_classifier(customer_features)
    )
    customer_segment_and_risk = customer_segments.merge(
        decline_scoring[
            [
                "Customer_ID",
                "decline_risk_score",
                "top_shap_driver_1",
                "top_shap_driver_2",
                "top_shap_driver_3",
                "top_shap_impact_1",
                "top_shap_impact_2",
                "top_shap_impact_3",
            ]
        ],
        on="Customer_ID",
        how="left",
        validate="one_to_one",
    ).sort_values("Customer_ID", ignore_index=True)

    cleaned_transactions.to_csv(CLEANED_OUTPUT_PATH, index=False)
    sow_eligible_transactions.to_csv(SOW_WINDOW_OUTPUT_PATH, index=False)
    lifetime_sow.to_csv(LIFETIME_SOW_OUTPUT_PATH, index=False)
    half_window_sow.to_csv(HALF_WINDOW_SOW_OUTPUT_PATH, index=False)
    monthly_sow.to_csv(MONTHLY_SOW_OUTPUT_PATH, index=False)
    quarterly_sow.to_csv(QUARTERLY_SOW_OUTPUT_PATH, index=False)
    customer_features.to_csv(CUSTOMER_FEATURES_OUTPUT_PATH, index=False)
    kmeans_evaluation.to_csv(KMEANS_EVALUATION_OUTPUT_PATH, index=False)
    segment_profiles.to_csv(SEGMENT_PROFILE_OUTPUT_PATH, index=False)
    segment_strategy.to_csv(SEGMENT_STRATEGY_OUTPUT_PATH, index=False)
    customer_segment_and_risk.to_csv(SEGMENTED_CUSTOMERS_OUTPUT_PATH, index=False)
    logistic_coefficients.to_csv(LOGISTIC_COEFFICIENTS_OUTPUT_PATH, index=False)
    calibration_curve_data.to_csv(CALIBRATION_CURVE_OUTPUT_PATH, index=False)
    with DECLINE_MODEL_METRICS_OUTPUT_PATH.open("w", encoding="ascii") as metrics_file:
        json.dump(decline_metrics, metrics_file, indent=2)

    lifetime_positive_total_count = int(lifetime_sow["lifetime_total_spend"].gt(0).sum())
    lifetime_out_of_bounds_count = int(lifetime_sow["sow_out_of_bounds"].sum())
    lifetime_out_of_bounds_share = (
        lifetime_out_of_bounds_count / lifetime_positive_total_count
        if lifetime_positive_total_count
        else 0.0
    )
    quarterly_valid_count = int(quarterly_sow["SoW_quarter"].notna().sum())
    quarterly_out_of_bounds_count = int(quarterly_sow["sow_out_of_bounds"].sum())
    quarterly_out_of_bounds_share = (
        quarterly_out_of_bounds_count / quarterly_valid_count
        if quarterly_valid_count
        else 0.0
    )
    selected_k = int(kmeans_evaluation.loc[kmeans_evaluation["selected_k"], "k"].iloc[0])
    risk_scored_customers = int(customer_segment_and_risk["decline_risk_score"].notna().sum())

    print(f"Cleaned dataset saved to: {CLEANED_OUTPUT_PATH}")
    print(f"SoW-eligible dataset saved to: {SOW_WINDOW_OUTPUT_PATH}")
    print(f"Lifetime SoW dataset saved to: {LIFETIME_SOW_OUTPUT_PATH}")
    print(f"Half-window SoW dataset saved to: {HALF_WINDOW_SOW_OUTPUT_PATH}")
    print(f"Monthly diagnostic dataset saved to: {MONTHLY_SOW_OUTPUT_PATH}")
    print(f"Quarterly diagnostic dataset saved to: {QUARTERLY_SOW_OUTPUT_PATH}")
    print(f"Customer feature table saved to: {CUSTOMER_FEATURES_OUTPUT_PATH}")
    print(f"KMeans evaluation saved to: {KMEANS_EVALUATION_OUTPUT_PATH}")
    print(f"Segment profile table saved to: {SEGMENT_PROFILE_OUTPUT_PATH}")
    print(f"Segment strategy table saved to: {SEGMENT_STRATEGY_OUTPUT_PATH}")
    print(f"Customer segment and risk table saved to: {SEGMENTED_CUSTOMERS_OUTPUT_PATH}")
    print(f"Logistic coefficients saved to: {LOGISTIC_COEFFICIENTS_OUTPUT_PATH}")
    print(f"Calibration curve saved to: {CALIBRATION_CURVE_OUTPUT_PATH}")
    print(f"Decline model metrics saved to: {DECLINE_MODEL_METRICS_OUTPUT_PATH}")
    print(f"SHAP summary plot saved to: {SHAP_SUMMARY_OUTPUT_PATH}")
    print(f"Study end date: {study_end_date.date()}")
    print(f"Customers dropped for missing open date: {dropped_customer_count:,}")
    print(f"Transaction rows dropped by customer inner join: {dropped_transaction_count:,}")
    print(f"Rows after customer join: {len(cleaned_transactions):,}")
    print(f"Rows in active card period: {len(sow_eligible_transactions):,}")
    print(f"Customer lifetime rows: {len(lifetime_sow):,}")
    print(
        "Lifetime customers usable for bounded features: "
        f"{int(lifetime_sow['use_for_bounded_features'].sum()):,}"
    )
    print(
        "Lifetime out-of-bounds share: "
        f"{lifetime_out_of_bounds_count:,}/{lifetime_positive_total_count:,} "
        f"({lifetime_out_of_bounds_share:.2%})"
    )
    print(
        "Lifetime ABC return-dominated customers: "
        f"{int(lifetime_sow['is_abc_return_dominated'].sum()):,}"
    )
    print(
        "Lifetime cross-channel return-skewed customers: "
        f"{int(lifetime_sow['is_cross_channel_return_skewed'].sum()):,}"
    )
    print(f"Customer half-window rows: {len(half_window_sow):,}")
    print(
        "Half-window delta usable for bounded features: "
        f"{int(half_window_sow['use_half_window_delta_for_bounded_features'].sum()):,}"
    )
    print(f"Customer fiscal-month rows: {len(monthly_sow):,}")
    print(f"Customer fiscal-quarter rows: {len(quarterly_sow):,}")
    print(f"Customer feature rows: {len(customer_features):,}")
    print(
        "Dormant cardholders in feature table: "
        f"{int(customer_features['is_dormant_cardholder'].sum()):,}"
    )
    print(
        "Customers with measurable lifetime SoW: "
        f"{int(customer_features['has_measurable_sow'].sum()):,}"
    )
    print(
        "Customers with usable half-window trend delta: "
        f"{int(customer_features['use_half_window_delta_for_bounded_features'].sum()):,}"
    )
    print(f"Selected K for measurable population: {selected_k}")
    print(f"Named segments produced: {customer_segments['cluster_segment'].nunique():,}")
    print(
        "Customers with decline risk scores: "
        f"{risk_scored_customers:,}"
    )
    print(
        "Logistic AUC / XGBoost AUC: "
        f"{decline_metrics['logistic_auc_roc']:.3f} / {decline_metrics['xgb_auc_roc']:.3f}"
    )
    print(
        "XGBoost top-decile precision: "
        f"{decline_metrics['xgb_top_decile_precision']:.3f}"
    )
    print(
        "Quarterly diagnostic out-of-bounds share: "
        f"{quarterly_out_of_bounds_count:,}/{quarterly_valid_count:,} "
        f"({quarterly_out_of_bounds_share:.2%})"
    )
    print(
        "ABC Bank Credit Card rows in active card period: "
        f"{int(sow_eligible_transactions['is_abc_bank_credit_card'].sum()):,}"
    )


if __name__ == "__main__":
    main()
