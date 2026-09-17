import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FINAL_RISK_FILE = (
    "data/processed/final_shortage_risk.csv"
)

PREDICTION_FILE = (
    "data/processed/shortage_risk_predictions.csv"
)

OUTPUT_DIR = "data/processed"

XAI_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "shortage_xai_explanations.csv"
)


# ============================================================
# EXACT STAGE 9 COLUMNS
# ============================================================

FINAL_RISK_COLUMNS = [
    "facility_id",
    "date",
    "ml_shortage_risk",
    "current_stockout",
    "spatial_shortage_pressure",
    "spatial_signal",
    "anomaly_evidence",
    "best_donor_facility_id",
    "best_donor_distance_km",
    "best_donor_surplus",
    "best_transit_probability",
    "best_feasibility_score",
    "best_feasibility_category",
    "regional_event",
    "final_shortage_score",
    "risk_level",
    "ml_high_risk_flag",
    "spatial_shortage_flag",
    "anomaly_flag",
    "redistribution_available",
    "evidence_count",
    "evidence_summary",
]


# ============================================================
# EXACT STAGE 5 COLUMNS REQUIRED FOR EXPLANATION
# ============================================================

PREDICTION_COLUMNS = [
    "facility_id",
    "date",
    "closing_inventory",
    "consumption",
    "received",
    "stockout",
    "stockout_risk",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def format_percentage(value):

    return f"{safe_float(value) * 100:.1f}%"


def format_number(value):

    value = safe_float(value)

    if value >= 1000:

        return f"{value:,.0f}"

    return f"{value:.1f}"


# ============================================================
# BUILD EXPLANATION
# ============================================================

def build_explanation(row):

    reasons = []

    observed_evidence = []

    model_evidence = []

    decision_evidence = []

    # --------------------------------------------------------
    # ML RISK
    # --------------------------------------------------------

    ml_risk = safe_float(
        row["ml_shortage_risk"]
    )

    if ml_risk >= 0.70:

        reason = (
            f"High ML-predicted shortage risk "
            f"({format_percentage(ml_risk)})"
        )

        reasons.append(reason)
        model_evidence.append(reason)

    elif ml_risk >= 0.50:

        reason = (
            f"Elevated ML-predicted shortage risk "
            f"({format_percentage(ml_risk)})"
        )

        reasons.append(reason)
        model_evidence.append(reason)

    else:

        model_evidence.append(
            f"ML-predicted shortage risk "
            f"is {format_percentage(ml_risk)}"
        )

    # --------------------------------------------------------
    # CURRENT STOCKOUT
    # --------------------------------------------------------

    current_stockout = safe_float(
        row["current_stockout"]
    )

    if current_stockout >= 1:

        reason = (
            "Current stockout is observed in the dataset"
        )

        reasons.append(reason)
        observed_evidence.append(reason)

    # --------------------------------------------------------
    # SPATIAL SHORTAGE
    # --------------------------------------------------------

    spatial_pressure = safe_float(
        row["spatial_shortage_pressure"]
    )

    if spatial_pressure >= 0.60:

        reason = (
            f"High nearby shortage pressure "
            f"({format_percentage(spatial_pressure)})"
        )

        reasons.append(reason)
        model_evidence.append(reason)

    elif spatial_pressure >= 0.40:

        reason = (
            f"Elevated nearby shortage pressure "
            f"({format_percentage(spatial_pressure)})"
        )

        reasons.append(reason)
        model_evidence.append(reason)

    # --------------------------------------------------------
    # SPATIAL SIGNAL
    # --------------------------------------------------------

    spatial_signal = safe_float(
        row["spatial_signal"]
    )

    if spatial_signal >= 1:

        observed_evidence.append(
            "A regional spatial shortage signal is present"
        )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    anomaly_evidence = safe_float(
        row["anomaly_evidence"]
    )

    anomaly_flag = safe_float(
        row["anomaly_flag"]
    )

    if anomaly_flag >= 1:

        reason = (
            f"Unusual consumption/inventory behavior "
            f"was detected "
            f"(anomaly evidence {anomaly_evidence:.3f})"
        )

        reasons.append(reason)
        model_evidence.append(reason)

    # --------------------------------------------------------
    # REGIONAL EVENT
    # --------------------------------------------------------

    regional_event = safe_float(
        row["regional_event"]
    )

    if regional_event >= 1:

        reason = (
            "The facility belongs to a period with "
            "a detected regional shortage event"
        )

        reasons.append(reason)
        model_evidence.append(reason)

    # --------------------------------------------------------
    # REDISTRIBUTION
    # --------------------------------------------------------

    redistribution_available = safe_float(
        row["redistribution_available"]
    )

    if redistribution_available >= 1:

        donor_id = row[
            "best_donor_facility_id"
        ]

        distance = safe_float(
            row["best_donor_distance_km"]
        )

        surplus = safe_float(
            row["best_donor_surplus"]
        )

        transit = safe_float(
            row["best_transit_probability"]
        )

        feasibility = safe_float(
            row["best_feasibility_score"]
        )

        category = str(
            row["best_feasibility_category"]
        )

        reason = (
            f"Redistribution option available from "
            f"facility {donor_id} "
            f"({distance:.2f} km away, "
            f"surplus {format_number(surplus)}, "
            f"feasibility {feasibility:.3f}, "
            f"category {category})"
        )

        reasons.append(reason)
        decision_evidence.append(reason)

        decision_evidence.append(
            f"Simulated transit feasibility probability "
            f"is {format_percentage(transit)}"
        )

    else:

        decision_evidence.append(
            "No redistribution option is available "
            "in the current simulated donor network"
        )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_score = safe_float(
        row["final_shortage_score"]
    )

    risk_level = str(
        row["risk_level"]
    )

    # --------------------------------------------------------
    # OVERALL EXPLANATION
    # --------------------------------------------------------

    if reasons:

        headline = (
            f"{risk_level} shortage alert: "
            f"facility {row['facility_id']} "
            f"has a fused shortage score of "
            f"{final_score:.3f}."
        )

        explanation = (
            headline
            + " "
            + " ".join(reasons)
        )

    else:

        explanation = (
            f"{risk_level} shortage assessment: "
            f"facility {row['facility_id']} "
            f"has a fused shortage score of "
            f"{final_score:.3f}, with limited "
            f"supporting evidence."
        )

    # --------------------------------------------------------
    # EVIDENCE COUNTS
    # --------------------------------------------------------

    evidence_count = len(reasons)

    # --------------------------------------------------------
    # RETURN STRUCTURED XAI
    # --------------------------------------------------------

    return pd.Series(
        {
            "xai_explanation": explanation,

            "xai_evidence_count": evidence_count,

            "xai_observed_evidence":
                " | ".join(observed_evidence),

            "xai_model_evidence":
                " | ".join(model_evidence),

            "xai_decision_evidence":
                " | ".join(decision_evidence),

            "xai_primary_reason":
                reasons[0]
                if reasons
                else "No dominant reason identified",

            "xai_final_score":
                final_score,

            "xai_risk_level":
                risk_level,
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - XAI EXPLANATION ENGINE")
    print("=" * 75)

    # ========================================================
    # 1. LOAD FINAL RISK
    # ========================================================

    print("\n" + "=" * 75)
    print("1. LOADING FINAL RISK OUTPUT")
    print("=" * 75)

    if not os.path.exists(FINAL_RISK_FILE):

        raise FileNotFoundError(
            f"Final risk file not found: "
            f"{FINAL_RISK_FILE}"
        )

    final_risk = pd.read_csv(
        FINAL_RISK_FILE
    )

    print(
        f"Final risk records: "
        f"{len(final_risk):,}"
    )

    print(
        f"Columns: "
        f"{list(final_risk.columns)}"
    )

    # ========================================================
    # 2. EXACT COLUMN CHECK
    # ========================================================

    print("\n" + "=" * 75)
    print("2. VALIDATING FINAL RISK COLUMNS")
    print("=" * 75)

    missing_final = [
        column
        for column in FINAL_RISK_COLUMNS
        if column not in final_risk.columns
    ]

    if missing_final:

        print(
            "Missing columns:"
        )

        for column in missing_final:

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Stage 9 column mismatch detected."
        )

    print(
        "All Stage 9 columns found."
    )

    print(
        "Column-name check: PASS"
    )

    # ========================================================
    # 3. LOAD PREDICTIONS
    # ========================================================

    print("\n" + "=" * 75)
    print("3. LOADING ML PREDICTIONS")
    print("=" * 75)

    if not os.path.exists(PREDICTION_FILE):

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    print(
        f"Prediction records: "
        f"{len(predictions):,}"
    )

    print(
        f"Columns: "
        f"{list(predictions.columns)}"
    )

    missing_prediction = [
        column
        for column in PREDICTION_COLUMNS
        if column not in predictions.columns
    ]

    if missing_prediction:

        print(
            "Missing prediction columns:"
        )

        for column in missing_prediction:

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Stage 5 prediction column mismatch detected."
        )

    print(
        "All required Stage 5 columns found."
    )

    print(
        "Prediction column-name check: PASS"
    )

    # ========================================================
    # 4. PREPARE JOIN
    # ========================================================

    print("\n" + "=" * 75)
    print("4. JOINING MODEL FEATURES")
    print("=" * 75)

    final_risk["date"] = pd.to_datetime(
        final_risk["date"],
        errors="coerce"
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="coerce"
    )

    prediction_subset = predictions[
        [
            "facility_id",
            "date",
            "closing_inventory",
            "consumption",
            "received",
            "stockout",
            "stockout_risk",
        ]
    ].copy()

    prediction_subset = (
        prediction_subset
        .rename(
            columns={
                "stockout_risk":
                    "prediction_stockout_risk",
            }
        )
    )

    merged = final_risk.merge(
        prediction_subset,
        on=[
            "facility_id",
            "date",
        ],
        how="left",
    )

    print(
        f"Merged records: "
        f"{len(merged):,}"
    )

    matched = int(
        merged[
            "prediction_stockout_risk"
        ]
        .notna()
        .sum()
    )

    print(
        f"Records with ML feature match: "
        f"{matched:,}"
    )

    # ========================================================
    # 5. GENERATE XAI
    # ========================================================

    print("\n" + "=" * 75)
    print("5. GENERATING EXPLANATIONS")
    print("=" * 75)

    xai = merged.apply(
        build_explanation,
        axis=1
    )

    output = pd.concat(
        [
            merged,
            xai,
        ],
        axis=1
    )

    print(
        f"Generated explanations: "
        f"{len(output):,}"
    )

    # ========================================================
    # 6. XAI SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("6. XAI SUMMARY")
    print("=" * 75)

    print(
        "\nRisk-level distribution:"
    )

    print(
        output[
            "xai_risk_level"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nAverage evidence count:"
    )

    print(
        f"{output['xai_evidence_count'].mean():.2f}"
    )

    print(
        "\nEvidence-count distribution:"
    )

    print(
        output[
            "xai_evidence_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # 7. OBSERVED / MODEL / DECISION EVIDENCE
    # ========================================================

    print("\n" + "=" * 75)
    print("7. EVIDENCE TYPE SUMMARY")
    print("=" * 75)

    observed_count = int(
        (
            output[
                "xai_observed_evidence"
            ]
            .str.len()
            > 0
        ).sum()
    )

    model_count = int(
        (
            output[
                "xai_model_evidence"
            ]
            .str.len()
            > 0
        ).sum()
    )

    decision_count = int(
        (
            output[
                "xai_decision_evidence"
            ]
            .str.len()
            > 0
        ).sum()
    )

    print(
        f"Records with observed evidence: "
        f"{observed_count:,}"
    )

    print(
        f"Records with model-derived evidence: "
        f"{model_count:,}"
    )

    print(
        f"Records with decision-support evidence: "
        f"{decision_count:,}"
    )

    # ========================================================
    # 8. DISPLAY TOP EXPLANATIONS
    # ========================================================

    print("\n" + "=" * 75)
    print("8. TOP EXPLANATIONS")
    print("=" * 75)

    top_explanations = (
        output
        .sort_values(
            "final_shortage_score",
            ascending=False
        )
        .head(10)
    )

    for _, row in top_explanations.iterrows():

        print(
            "\n------------------------------------------------------------"
        )

        print(
            f"Facility: {row['facility_id']}"
        )

        print(
            f"Date: {row['date'].date()}"
        )

        print(
            f"Risk level: {row['risk_level']}"
        )

        print(
            f"Final shortage score: "
            f"{safe_float(row['final_shortage_score']):.4f}"
        )

        print(
            f"Explanation:"
        )

        print(
            row["xai_explanation"]
        )

        print(
            f"Observed evidence:"
        )

        print(
            row["xai_observed_evidence"]
            if row["xai_observed_evidence"]
            else "None"
        )

        print(
            f"Model evidence:"
        )

        print(
            row["xai_model_evidence"]
            if row["xai_model_evidence"]
            else "None"
        )

        print(
            f"Decision evidence:"
        )

        print(
            row["xai_decision_evidence"]
            if row["xai_decision_evidence"]
            else "None"
        )

    # ========================================================
    # 9. SAVE
    # ========================================================

    print("\n" + "=" * 75)
    print("9. SAVING XAI OUTPUT")
    print("=" * 75)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output.to_csv(
        XAI_OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved: {XAI_OUTPUT_FILE}"
    )

    # ========================================================
    # 10. FINAL STATUS
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 12 COMPLETE")
    print("=" * 75)

    print(
        "Evidence-based XAI explanations were generated "
        "for the final shortage-risk records."
    )

    print(
        "\nThe explanations distinguish:"
    )

    print(
        "1. Observed evidence from the real dataset."
    )

    print(
        "2. Model-derived evidence."
    )

    print(
        "3. Simulated decision-support evidence."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The explanation describes the evidence used "
        "by the prototype. It does not establish "
        "causation."
    )

    print(
        "The final shortage score is an evidence-fusion "
        "score, not a calibrated probability."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()