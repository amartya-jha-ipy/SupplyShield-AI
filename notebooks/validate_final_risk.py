import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

FINAL_RISK_FILE = "data/processed/final_shortage_risk.csv"

PREDICTION_FILE = "data/processed/shortage_risk_predictions.csv"

OUTPUT_DIR = "data/processed"

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "final_risk_validation_summary.csv"
)

THRESHOLD_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "final_risk_threshold_evaluation.csv"
)

TOP_RISK_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "final_risk_top_percentile_evaluation.csv"
)

MONTHLY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "final_risk_monthly_validation.csv"
)


# ============================================================
# HELPERS
# ============================================================

def print_section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def find_column(df, candidates):

    normalized = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = str(candidate).lower()

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - FINAL RISK VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD FINAL RISK
    # --------------------------------------------------------

    print_section(
        "1. LOADING FINAL RISK OUTPUT"
    )

    if not os.path.exists(FINAL_RISK_FILE):

        raise FileNotFoundError(
            f"Final risk file not found: "
            f"{FINAL_RISK_FILE}"
        )

    final_df = pd.read_csv(
        FINAL_RISK_FILE
    )

    print(
        f"Final-risk records: "
        f"{len(final_df):,}"
    )

    print(
        f"Final-risk columns: "
        f"{list(final_df.columns)}"
    )

    # --------------------------------------------------------
    # 2. LOAD ACTUAL FUTURE-STOCKOUT TARGET
    # --------------------------------------------------------

    print_section(
        "2. LOADING OBSERVED FUTURE-STOCKOUT TARGET"
    )

    if not os.path.exists(PREDICTION_FILE):

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{PREDICTION_FILE}"
        )

    prediction_df = pd.read_csv(
        PREDICTION_FILE
    )

    print(
        f"Prediction records: "
        f"{len(prediction_df):,}"
    )

    print(
        f"Prediction columns: "
        f"{list(prediction_df.columns)}"
    )

    # --------------------------------------------------------
    # 3. IDENTIFY FINAL-RISK COLUMNS
    # --------------------------------------------------------

    print_section(
        "3. IDENTIFYING FINAL-RISK COLUMNS"
    )

    final_facility_col = find_column(
        final_df,
        [
            "facility_id",
            "hf_pk",
        ]
    )

    final_date_col = find_column(
        final_df,
        [
            "date",
        ]
    )

    final_score_col = find_column(
        final_df,
        [
            "final_shortage_score",
            "final_risk_score",
            "shortage_score",
            "risk_score",
        ]
    )

    if final_facility_col is None:

        raise ValueError(
            "Could not identify facility ID column "
            "in final risk file."
        )

    if final_date_col is None:

        raise ValueError(
            "Could not identify date column "
            "in final risk file."
        )

    if final_score_col is None:

        raise ValueError(
            "Could not identify final shortage "
            "risk score."
        )

    print(
        f"Facility column: "
        f"{final_facility_col}"
    )

    print(
        f"Date column: "
        f"{final_date_col}"
    )

    print(
        f"Final score column: "
        f"{final_score_col}"
    )

    # --------------------------------------------------------
    # 4. IDENTIFY PREDICTION TARGET COLUMNS
    # --------------------------------------------------------

    print_section(
        "4. IDENTIFYING OBSERVED TARGET COLUMNS"
    )

    prediction_facility_col = find_column(
        prediction_df,
        [
            "facility_id",
            "hf_pk",
        ]
    )

    prediction_date_col = find_column(
        prediction_df,
        [
            "date",
        ]
    )

    target_col = find_column(
        prediction_df,
        [
            "target_next_stockout",
            "next_stockout",
            "target_stockout",
        ]
    )

    if prediction_facility_col is None:

        raise ValueError(
            "Could not identify facility ID column "
            "in prediction file."
        )

    if prediction_date_col is None:

        raise ValueError(
            "Could not identify date column "
            "in prediction file."
        )

    if target_col is None:

        raise ValueError(
            "Could not identify target_next_stockout "
            "column in prediction file."
        )

    print(
        f"Prediction facility column: "
        f"{prediction_facility_col}"
    )

    print(
        f"Prediction date column: "
        f"{prediction_date_col}"
    )

    print(
        f"Observed target column: "
        f"{target_col}"
    )

    # --------------------------------------------------------
    # 5. PREPARE JOIN KEYS
    # --------------------------------------------------------

    print_section(
        "5. PREPARING FINAL-RISK / TARGET JOIN"
    )

    final_df[final_date_col] = pd.to_datetime(
        final_df[final_date_col],
        errors="coerce"
    )

    prediction_df[prediction_date_col] = pd.to_datetime(
        prediction_df[prediction_date_col],
        errors="coerce"
    )

    final_df[final_facility_col] = pd.to_numeric(
        final_df[final_facility_col],
        errors="coerce"
    )

    prediction_df[prediction_facility_col] = pd.to_numeric(
        prediction_df[prediction_facility_col],
        errors="coerce"
    )

    final_df[final_score_col] = pd.to_numeric(
        final_df[final_score_col],
        errors="coerce"
    )

    prediction_df[target_col] = pd.to_numeric(
        prediction_df[target_col],
        errors="coerce"
    )

    final_df = final_df.dropna(
        subset=[
            final_facility_col,
            final_date_col,
            final_score_col,
        ]
    ).copy()

    prediction_df = prediction_df.dropna(
        subset=[
            prediction_facility_col,
            prediction_date_col,
            target_col,
        ]
    ).copy()

    prediction_df = prediction_df[
        prediction_df[target_col].isin(
            [0, 1]
        )
    ].copy()

    prediction_df[target_col] = (
        prediction_df[target_col]
        .astype(int)
    )

    # --------------------------------------------------------
    # 6. RENAME JOIN COLUMNS
    # --------------------------------------------------------

    final_join = final_df[
        [
            final_facility_col,
            final_date_col,
            final_score_col,
        ]
    ].copy()

    final_join = final_join.rename(
        columns={
            final_facility_col: "facility_id",
            final_date_col: "date",
            final_score_col: "final_shortage_score",
        }
    )

    prediction_join = prediction_df[
        [
            prediction_facility_col,
            prediction_date_col,
            target_col,
        ]
    ].copy()

    prediction_join = prediction_join.rename(
        columns={
            prediction_facility_col: "facility_id",
            prediction_date_col: "date",
            target_col: "target_next_stockout",
        }
    )

    # Remove duplicate facility-date keys if any.

    final_join = (
        final_join
        .drop_duplicates(
            subset=[
                "facility_id",
                "date",
            ]
        )
    )

    prediction_join = (
        prediction_join
        .drop_duplicates(
            subset=[
                "facility_id",
                "date",
            ]
        )
    )

    print(
        f"Final-risk unique facility-date records: "
        f"{len(final_join):,}"
    )

    print(
        f"Target unique facility-date records: "
        f"{len(prediction_join):,}"
    )

    # --------------------------------------------------------
    # 7. JOIN
    # --------------------------------------------------------

    print_section(
        "7. JOINING FINAL RISK WITH OBSERVED OUTCOMES"
    )

    df = final_join.merge(
        prediction_join,
        on=[
            "facility_id",
            "date",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Matched validation records: "
        f"{len(df):,}"
    )

    if len(df) == 0:

        raise ValueError(
            "No final-risk records matched the "
            "observed target records."
        )

    print(
        f"Observed future stockouts: "
        f"{df['target_next_stockout'].sum():,}"
    )

    print(
        f"Observed future stockout rate: "
        f"{df['target_next_stockout'].mean():.2%}"
    )

    print(
        f"Final score range: "
        f"{df['final_shortage_score'].min():.4f} - "
        f"{df['final_shortage_score'].max():.4f}"
    )

    # --------------------------------------------------------
    # 8. OVERALL DISCRIMINATION
    # --------------------------------------------------------

    print_section(
        "8. OVERALL FINAL-RISK VALIDATION"
    )

    y_true = df[
        "target_next_stockout"
    ]

    y_score = df[
        "final_shortage_score"
    ]

    if y_true.nunique() < 2:

        raise ValueError(
            "Observed target contains fewer than "
            "two classes."
        )

    roc_auc = roc_auc_score(
        y_true,
        y_score
    )

    pr_auc = average_precision_score(
        y_true,
        y_score
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # 9. THRESHOLD EVALUATION
    # --------------------------------------------------------

    print_section(
        "9. THRESHOLD EVALUATION"
    )

    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    threshold_results = []

    for threshold in thresholds:

        predicted = (
            y_score >= threshold
        ).astype(int)

        alerts = int(
            predicted.sum()
        )

        precision = precision_score(
            y_true,
            predicted,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predicted,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predicted,
            zero_division=0
        )

        cm = confusion_matrix(
            y_true,
            predicted,
            labels=[
                0,
                1,
            ]
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "alerts": alerts,
                "alert_rate": (
                    alerts / len(df)
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"alerts={alerts:,} | "
            f"precision={precision:.3f} | "
            f"recall={recall:.3f} | "
            f"F1={f1:.3f}"
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # --------------------------------------------------------
    # 10. TOP-RISK GROUP VALIDATION
    # --------------------------------------------------------

    print_section(
        "10. TOP-RISK GROUP VALIDATION"
    )

    top_percentages = [
        0.05,
        0.10,
        0.20,
        0.30,
    ]

    baseline_rate = y_true.mean()

    top_results = []

    sorted_df = (
        df
        .sort_values(
            "final_shortage_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    for percentage in top_percentages:

        count = max(
            1,
            int(
                np.ceil(
                    len(sorted_df)
                    * percentage
                )
            )
        )

        top_group = (
            sorted_df
            .head(count)
        )

        top_stockout_rate = (
            top_group[
                "target_next_stockout"
            ]
            .mean()
        )

        lift = (
            top_stockout_rate
            / baseline_rate
            if baseline_rate > 0
            else np.nan
        )

        top_results.append(
            {
                "top_percentage": percentage,
                "records": count,
                "stockouts": int(
                    top_group[
                        "target_next_stockout"
                    ].sum()
                ),
                "stockout_rate": (
                    top_stockout_rate
                ),
                "baseline_stockout_rate": (
                    baseline_rate
                ),
                "lift": lift,
                "average_score": (
                    top_group[
                        "final_shortage_score"
                    ].mean()
                ),
            }
        )

        print(
            f"Top {percentage:.0%}: "
            f"{count:,} records | "
            f"stockout rate="
            f"{top_stockout_rate:.2%} | "
            f"lift="
            f"{lift:.2f}x"
        )

    top_df = pd.DataFrame(
        top_results
    )

    # --------------------------------------------------------
    # 11. SCORE QUANTILES
    # --------------------------------------------------------

    print_section(
        "11. STOCKOUT RATE ACROSS FINAL-SCORE QUANTILES"
    )

    try:

        df["risk_quantile"] = pd.qcut(
            df["final_shortage_score"],
            q=5,
            labels=[
                "Q1 lowest",
                "Q2",
                "Q3",
                "Q4",
                "Q5 highest",
            ],
            duplicates="drop"
        )

        quantile_summary = (
            df.groupby(
                "risk_quantile",
                observed=False
            )
            .agg(
                records=(
                    "target_next_stockout",
                    "count"
                ),
                stockouts=(
                    "target_next_stockout",
                    "sum"
                ),
                stockout_rate=(
                    "target_next_stockout",
                    "mean"
                ),
                average_score=(
                    "final_shortage_score",
                    "mean"
                ),
            )
            .reset_index()
        )

        print(
            quantile_summary.to_string(
                index=False
            )
        )

    except ValueError:

        print(
            "Could not create final-risk quantiles."
        )

        quantile_summary = pd.DataFrame()

    # --------------------------------------------------------
    # 12. MONTHLY VALIDATION
    # --------------------------------------------------------

    print_section(
        "12. MONTHLY FINAL-RISK VALIDATION"
    )

    monthly = (
        df.groupby(
            "date"
        )
        .agg(
            records=(
                "target_next_stockout",
                "count"
            ),
            observed_future_stockouts=(
                "target_next_stockout",
                "sum"
            ),
            future_stockout_rate=(
                "target_next_stockout",
                "mean"
            ),
            average_final_score=(
                "final_shortage_score",
                "mean"
            ),
            maximum_final_score=(
                "final_shortage_score",
                "max"
            ),
        )
        .reset_index()
    )

    print(
        monthly.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 13. HIGH-RISK ANALYSIS
    # --------------------------------------------------------

    print_section(
        "13. HIGH-RISK ALERT ANALYSIS"
    )

    high_threshold = 0.70

    high_risk = (
        df["final_shortage_score"]
        >= high_threshold
    )

    high_risk_count = int(
        high_risk.sum()
    )

    if high_risk_count > 0:

        high_risk_rate = (
            df.loc[
                high_risk,
                "target_next_stockout"
            ]
            .mean()
        )

    else:

        high_risk_rate = np.nan

    print(
        f"High-risk threshold: "
        f"{high_threshold:.2f}"
    )

    print(
        f"High-risk records: "
        f"{high_risk_count:,}"
    )

    if not np.isnan(high_risk_rate):

        high_risk_lift = (
            high_risk_rate
            / baseline_rate
        )

        print(
            f"High-risk future-stockout rate: "
            f"{high_risk_rate:.2%}"
        )

        print(
            f"High-risk lift: "
            f"{high_risk_lift:.2f}x"
        )

    else:

        high_risk_lift = np.nan

        print(
            "No records crossed the high-risk threshold."
        )

    # --------------------------------------------------------
    # 14. CONFUSION MATRIX
    # --------------------------------------------------------

    print_section(
        "14. CONFUSION MATRIX AT THRESHOLD 0.50"
    )

    predictions_50 = (
        y_score >= 0.50
    ).astype(int)

    cm_50 = confusion_matrix(
        y_true,
        predictions_50,
        labels=[
            0,
            1,
        ]
    )

    print(
        "                 Predicted"
    )

    print(
        "                 0       1"
    )

    print(
        f"Actual 0       "
        f"{cm_50[0, 0]:7d} "
        f"{cm_50[0, 1]:7d}"
    )

    print(
        f"Actual 1       "
        f"{cm_50[1, 0]:7d} "
        f"{cm_50[1, 1]:7d}"
    )

    # --------------------------------------------------------
    # 15. SAVE OUTPUTS
    # --------------------------------------------------------

    print_section(
        "15. SAVING VALIDATION OUTPUTS"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric": "matched_validation_records",
                "value": len(df),
            },
            {
                "metric": "observed_future_stockout_rate",
                "value": baseline_rate,
            },
            {
                "metric": "roc_auc",
                "value": roc_auc,
            },
            {
                "metric": "pr_auc",
                "value": pr_auc,
            },
            {
                "metric": "high_risk_threshold",
                "value": high_threshold,
            },
            {
                "metric": "high_risk_records",
                "value": high_risk_count,
            },
            {
                "metric": "high_risk_stockout_rate",
                "value": high_risk_rate,
            },
            {
                "metric": "high_risk_lift",
                "value": high_risk_lift,
            },
        ]
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    threshold_df.to_csv(
        THRESHOLD_OUTPUT,
        index=False
    )

    top_df.to_csv(
        TOP_RISK_OUTPUT,
        index=False
    )

    monthly.to_csv(
        MONTHLY_OUTPUT,
        index=False
    )

    print(
        f"Saved: {SUMMARY_OUTPUT}"
    )

    print(
        f"Saved: {THRESHOLD_OUTPUT}"
    )

    print(
        f"Saved: {TOP_RISK_OUTPUT}"
    )

    print(
        f"Saved: {MONTHLY_OUTPUT}"
    )

    # --------------------------------------------------------
    # 16. FINAL STATUS
    # --------------------------------------------------------

    print_section(
        "STAGE 11D COMPLETE"
    )

    print(
        f"Matched records: "
        f"{len(df):,}"
    )

    print(
        f"Final risk ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"Final risk PR-AUC: "
        f"{pr_auc:.4f}"
    )

    print(
        f"Baseline future-stockout rate: "
        f"{baseline_rate:.2%}"
    )

    if not np.isnan(high_risk_rate):

        print(
            f"High-risk future-stockout rate: "
            f"{high_risk_rate:.2%}"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The final shortage score is an evidence-fusion "
        "score, not a calibrated probability."
    )

    print(
        "The observed target comes from the actual "
        "next-month stockout outcome."
    )

    print(
        "This validation measures association with "
        "future observed stockouts."
    )

    print(
        "It does not establish causation or guarantee "
        "that a facility will experience a shortage."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()