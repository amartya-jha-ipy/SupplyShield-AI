import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_FILE = "data/processed/shortage_risk_predictions.csv"
TEST_FILE = "data/processed/shortage_risk_test.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model_validation_summary.csv")


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def find_column(df, candidates):
    """
    Return the first matching column from a list of candidates.
    Matching is case-insensitive.
    """
    normalized = {str(col).lower(): col for col in df.columns}

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
    print("SUPPLYSHIELD AI - MODEL VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD MODEL OUTPUT
    # --------------------------------------------------------

    print_section("1. LOADING MODEL PREDICTIONS")

    if not os.path.exists(PREDICTION_FILE):
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_FILE}"
        )

    predictions = pd.read_csv(PREDICTION_FILE)

    print(f"Prediction records: {len(predictions):,}")
    print(f"Prediction columns: {list(predictions.columns)}")

    # --------------------------------------------------------
    # 2. IDENTIFY IMPORTANT COLUMNS
    # --------------------------------------------------------

    risk_col = find_column(
        predictions,
        [
            "stockout_risk",
            "shortage_probability",
            "predicted_probability",
            "risk_probability",
        ],
    )

    target_col = find_column(
        predictions,
        [
            "target_next_stockout",
            "next_stockout",
            "actual_next_stockout",
            "target",
        ],
    )

    date_col = find_column(
        predictions,
        [
            "date",
            "prediction_date",
        ],
    )

    facility_col = find_column(
        predictions,
        [
            "facility_id",
            "hf_pk",
        ],
    )

    if risk_col is None:
        raise ValueError(
            "Could not identify the model risk/probability column."
        )

    if target_col is None:
        raise ValueError(
            "Could not identify the actual next-month stockout target."
        )

    print(f"\nUsing risk column: {risk_col}")
    print(f"Using target column: {target_col}")

    # --------------------------------------------------------
    # 3. CLEAN VALIDATION DATA
    # --------------------------------------------------------

    print_section("2. PREPARING VALIDATION DATA")

    validation = predictions[
        [c for c in [facility_col, date_col, risk_col, target_col] if c is not None]
    ].copy()

    validation[risk_col] = pd.to_numeric(
        validation[risk_col],
        errors="coerce"
    )

    validation[target_col] = pd.to_numeric(
        validation[target_col],
        errors="coerce"
    )

    validation = validation.dropna(
        subset=[risk_col, target_col]
    )

    validation[target_col] = validation[target_col].astype(int)

    # Keep only valid binary targets.
    validation = validation[
        validation[target_col].isin([0, 1])
    ].copy()

    print(f"Valid validation records: {len(validation):,}")

    if len(validation) == 0:
        raise ValueError("No valid validation records remain.")

    y_true = validation[target_col].values
    y_score = validation[risk_col].values

    print(
        f"Actual next-month stockouts: "
        f"{int(y_true.sum()):,}"
    )

    print(
        f"Actual next-month non-stockouts: "
        f"{int((y_true == 0).sum()):,}"
    )

    print(
        f"Actual stockout rate: "
        f"{y_true.mean():.2%}"
    )

    print(
        f"Prediction risk range: "
        f"{y_score.min():.4f} - {y_score.max():.4f}"
    )

    # --------------------------------------------------------
    # 4. ROC-AUC
    # --------------------------------------------------------

    print_section("3. ROC-AUC")

    if len(np.unique(y_true)) < 2:
        print("ROC-AUC cannot be calculated because only one class exists.")
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(
            y_true,
            y_score
        )

        print(f"ROC-AUC: {roc_auc:.4f}")

    # --------------------------------------------------------
    # 5. PR-AUC
    # --------------------------------------------------------

    print_section("4. PRECISION-RECALL AUC")

    if len(np.unique(y_true)) < 2:
        print("PR-AUC cannot be calculated because only one class exists.")
        pr_auc = np.nan
    else:
        pr_auc = average_precision_score(
            y_true,
            y_score
        )

        print(f"PR-AUC: {pr_auc:.4f}")

    # --------------------------------------------------------
    # 6. THRESHOLD EVALUATION
    # --------------------------------------------------------

    print_section("5. THRESHOLD EVALUATION")

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

        y_pred = (
            y_score >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        predicted_alerts = int(
            y_pred.sum()
        )

        true_positives = int(
            ((y_pred == 1) & (y_true == 1)).sum()
        )

        false_positives = int(
            ((y_pred == 1) & (y_true == 0)).sum()
        )

        false_negatives = int(
            ((y_pred == 0) & (y_true == 1)).sum()
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "predicted_alerts": predicted_alerts,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"alerts={predicted_alerts:4d} | "
            f"precision={precision:.3f} | "
            f"recall={recall:.3f} | "
            f"F1={f1:.3f}"
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # --------------------------------------------------------
    # 7. DETAILED REPORT AT 0.50
    # --------------------------------------------------------

    print_section("6. CLASSIFICATION REPORT AT THRESHOLD 0.50")

    default_threshold = 0.50

    y_pred_default = (
        y_score >= default_threshold
    ).astype(int)

    print(
        classification_report(
            y_true,
            y_pred_default,
            target_names=[
                "No Stockout",
                "Stockout",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred_default
        )
    )

    # --------------------------------------------------------
    # 8. TOP-RISK PRECISION
    # --------------------------------------------------------

    print_section("7. TOP-RISK FACILITY EVALUATION")

    top_fraction_results = []

    for fraction in [0.05, 0.10, 0.20]:

        count = max(
            1,
            int(
                np.ceil(
                    len(validation) * fraction
                )
            ),
        )

        ranked = validation.sort_values(
            risk_col,
            ascending=False
        )

        top = ranked.head(count)

        top_actual_stockout_rate = (
            top[target_col].mean()
        )

        overall_stockout_rate = (
            validation[target_col].mean()
        )

        lift = (
            top_actual_stockout_rate
            / overall_stockout_rate
            if overall_stockout_rate > 0
            else np.nan
        )

        top_fraction_results.append(
            {
                "top_fraction": fraction,
                "records": count,
                "actual_stockout_rate": top_actual_stockout_rate,
                "overall_stockout_rate": overall_stockout_rate,
                "lift": lift,
            }
        )

        print(
            f"Top {fraction:.0%} of predictions | "
            f"records={count} | "
            f"actual stockout rate="
            f"{top_actual_stockout_rate:.2%} | "
            f"lift={lift:.2f}x"
        )

    top_fraction_df = pd.DataFrame(
        top_fraction_results
    )

    # --------------------------------------------------------
    # 9. SAVE SUMMARY
    # --------------------------------------------------------

    print_section("8. SAVING VALIDATION OUTPUT")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summary_rows = [
        {
            "metric": "validation_records",
            "value": len(validation),
        },
        {
            "metric": "actual_stockout_rate",
            "value": y_true.mean(),
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
            "metric": "threshold_0.50_precision",
            "value": precision_score(
                y_true,
                y_pred_default,
                zero_division=0
            ),
        },
        {
            "metric": "threshold_0.50_recall",
            "value": recall_score(
                y_true,
                y_pred_default,
                zero_division=0
            ),
        },
        {
            "metric": "threshold_0.50_f1",
            "value": f1_score(
                y_true,
                y_pred_default,
                zero_division=0
            ),
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # 10. SAVE THRESHOLD RESULTS
    # --------------------------------------------------------

    threshold_output = os.path.join(
        OUTPUT_DIR,
        "model_threshold_evaluation.csv"
    )

    threshold_df.to_csv(
        threshold_output,
        index=False
    )

    print(
        f"Saved: {threshold_output}"
    )

    # --------------------------------------------------------
    # 11. SAVE TOP-FRACTION RESULTS
    # --------------------------------------------------------

    top_fraction_output = os.path.join(
        OUTPUT_DIR,
        "model_top_risk_evaluation.csv"
    )

    top_fraction_df.to_csv(
        top_fraction_output,
        index=False
    )

    print(
        f"Saved: {top_fraction_output}"
    )

    print_section("STAGE 11A COMPLETE")

    print(
        "The model has now been evaluated against "
        "real next-month stockout outcomes."
    )


if __name__ == "__main__":
    main()