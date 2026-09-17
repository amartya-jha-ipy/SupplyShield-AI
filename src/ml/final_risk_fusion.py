"""
SupplyShield AI
Stage 9 - Final Shortage Risk / Evidence Fusion

Purpose:
Combine the independent signals produced by the previous stages
into one transparent decision-support output.

Inputs:
1. Stage 5:
   shortage_risk_predictions.csv

2. Stage 6:
   spatial_shortage_signals.csv
   regional_shortage_events.csv

3. Stage 7:
   anomaly_signals.csv

4. Stage 8:
   redistribution_feasibility.csv

Important:
The final score is an evidence-fusion score.
It is NOT a newly trained probability model.

The real dataset does not contain observed supplier relationships
or observed transportation lead times. Therefore, redistribution
feasibility remains based on explicitly simulated assumptions.

The purpose of this stage is to provide:
- facility-level risk
- supporting evidence
- urgency
- possible redistribution options
- explainable alerts
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# INPUT FILES
# ============================================================

PREDICTION_FILE = Path(
    "data/processed/shortage_risk_predictions.csv"
)

SPATIAL_FILE = Path(
    "data/processed/spatial_shortage_signals.csv"
)

REGIONAL_FILE = Path(
    "data/processed/regional_shortage_events.csv"
)

ANOMALY_FILE = Path(
    "data/processed/anomaly_signals.csv"
)

FEASIBILITY_FILE = Path(
    "data/processed/redistribution_feasibility.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

FINAL_RISK_FILE = Path(
    "data/processed/final_shortage_risk.csv"
)

FINAL_ALERT_FILE = Path(
    "data/processed/final_alerts.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Weight assigned to the ML shortage-risk model.
ML_WEIGHT = 0.45

# Weight assigned to spatial shortage evidence.
SPATIAL_WEIGHT = 0.20

# Weight assigned to behavioural anomaly evidence.
ANOMALY_WEIGHT = 0.20

# Weight assigned to redistribution evidence.
FEASIBILITY_WEIGHT = 0.15


# Risk thresholds for the final evidence-fusion score.

HIGH_RISK_THRESHOLD = 0.75

MODERATE_RISK_THRESHOLD = 0.50

LOW_RISK_THRESHOLD = 0.25


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(
    df,
    candidates,
    required=True
):
    """
    Return the first matching column from candidates.
    """

    for column in candidates:

        if column in df.columns:
            return column

    if required:

        raise ValueError(
            "Could not find any of these columns: "
            + ", ".join(candidates)
        )

    return None


def normalize_series(series):
    """
    Min-max normalize a numeric series to [0, 1].

    Constant series become zero.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum):

        return pd.Series(
            0.0,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (values - minimum)
        /
        (maximum - minimum)
    ).clip(
        0.0,
        1.0
    )


def classify_risk(score):
    """
    Convert final score to an interpretable risk level.
    """

    if score >= HIGH_RISK_THRESHOLD:
        return "HIGH"

    if score >= MODERATE_RISK_THRESHOLD:
        return "MODERATE"

    if score >= LOW_RISK_THRESHOLD:
        return "LOW"

    return "VERY_LOW"


def build_evidence_text(row):
    """
    Build a short human-readable explanation of why
    a facility received its risk score.
    """

    evidence = []

    if row["ml_shortage_risk"] >= 0.50:

        evidence.append(
            "high ML shortage risk"
        )

    if row["spatial_shortage_pressure"] >= 0.50:

        evidence.append(
            "nearby facilities show shortage pressure"
        )

    if row["anomaly_evidence"] >= 0.50:

        evidence.append(
            "unusual demand/inventory behaviour"
        )

    if row["best_feasibility_score"] >= 0.50:

        evidence.append(
            "redistribution option available"
        )

    if row["current_stockout"] == 1:

        evidence.append(
            "currently observed stockout"
        )

    if not evidence:

        evidence.append(
            "limited shortage evidence"
        )

    return "; ".join(evidence)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - FINAL SHORTAGE RISK FUSION")
    print("=" * 75)

    # ========================================================
    # 1. LOAD STAGE 5
    # ========================================================

    print("\n" + "=" * 75)
    print("1. LOADING ML SHORTAGE-RISK SIGNAL")
    print("=" * 75)

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    print(
        f"Prediction records: "
        f"{len(predictions):,}"
    )

    required_prediction_columns = [
        "facility_id",
        "date",
        "stockout_risk",
        "stockout",
    ]

    missing = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing:

        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(missing)
        )

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="coerce"
    )

    predictions["stockout_risk"] = pd.to_numeric(
        predictions["stockout_risk"],
        errors="coerce"
    ).clip(
        0.0,
        1.0
    )

    predictions["stockout"] = pd.to_numeric(
        predictions["stockout"],
        errors="coerce"
    ).fillna(0).astype(int)

    # ========================================================
    # 2. LOAD STAGE 6 SPATIAL SIGNAL
    # ========================================================

    print("\n" + "=" * 75)
    print("2. LOADING SPATIAL SHORTAGE SIGNAL")
    print("=" * 75)

    if not SPATIAL_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {SPATIAL_FILE}"
        )

    spatial = pd.read_csv(
        SPATIAL_FILE
    )

    print(
        f"Spatial records: "
        f"{len(spatial):,}"
    )

    print(
        "Spatial columns:"
    )

    print(
        list(spatial.columns)
    )

    # Identify columns robustly.

    spatial_facility_column = find_column(
        spatial,
        [
            "facility_id",
            "hf_pk",
        ]
    )

    spatial_date_column = find_column(
        spatial,
        [
            "date",
        ]
    )

    spatial_pressure_column = find_column(
        spatial,
        [
            "local_shortage_pressure",
            "nearby_stockout_rate",
            "spatial_shortage_pressure",
        ],
        required=False
    )

    spatial_signal_column = find_column(
        spatial,
        [
            "regional_signal",
            "spatial_signal",
        ],
        required=False
    )

    spatial["date"] = pd.to_datetime(
        spatial[spatial_date_column],
        errors="coerce"
    )

    spatial = spatial.rename(
        columns={
            spatial_facility_column:
                "facility_id"
        }
    )

    if spatial_pressure_column is not None:

        spatial[
            "spatial_shortage_pressure"
        ] = normalize_series(
            spatial[
                spatial_pressure_column
            ]
        )

    else:

        spatial[
            "spatial_shortage_pressure"
        ] = 0.0

    if spatial_signal_column is not None:

        spatial[
            "spatial_signal"
        ] = pd.to_numeric(
            spatial[
                spatial_signal_column
            ],
            errors="coerce"
        ).fillna(0).astype(int)

    else:

        spatial[
            "spatial_signal"
        ] = 0

    spatial = spatial[
        [
            "facility_id",
            "date",
            "spatial_shortage_pressure",
            "spatial_signal",
        ]
    ]

    # ========================================================
    # 3. LOAD REGIONAL EVENTS
    # ========================================================

    print("\n" + "=" * 75)
    print("3. LOADING REGIONAL SHORTAGE EVENTS")
    print("=" * 75)

    if REGIONAL_FILE.exists():

        regional = pd.read_csv(
            REGIONAL_FILE
        )

        print(
            f"Regional summary records: "
            f"{len(regional):,}"
        )

        print(
            "Regional columns:"
        )

        print(
            list(regional.columns)
        )

    else:

        print(
            "Regional event file not found."
        )

        print(
            "Continuing without regional-event information."
        )

        regional = pd.DataFrame()

    # ========================================================
    # 4. LOAD STAGE 7 ANOMALIES
    # ========================================================

    print("\n" + "=" * 75)
    print("4. LOADING BEHAVIOURAL ANOMALIES")
    print("=" * 75)

    if not ANOMALY_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {ANOMALY_FILE}"
        )

    anomalies = pd.read_csv(
        ANOMALY_FILE
    )

    print(
        f"Anomaly records: "
        f"{len(anomalies):,}"
    )

    print(
        "Anomaly columns:"
    )

    print(
        list(anomalies.columns)
    )

    anomaly_facility_column = find_column(
        anomalies,
        [
            "facility_id",
            "hf_pk",
        ]
    )

    anomaly_date_column = find_column(
        anomalies,
        [
            "date",
        ]
    )

    anomaly_score_column = find_column(
        anomalies,
        [
            "combined_anomaly_score",
            "anomaly_score",
        ],
        required=False
    )

    shortage_anomaly_column = find_column(
        anomalies,
        [
            "shortage_relevant_anomaly",
            "shortage_anomaly",
        ],
        required=False
    )

    low_inventory_column = find_column(
        anomalies,
        [
            "low_inventory_anomaly",
            "low_inventory",
        ],
        required=False
    )

    high_consumption_column = find_column(
        anomalies,
        [
            "high_consumption_anomaly",
            "high_consumption",
        ],
        required=False
    )

    anomalies["date"] = pd.to_datetime(
        anomalies[anomaly_date_column],
        errors="coerce"
    )

    anomalies = anomalies.rename(
        columns={
            anomaly_facility_column:
                "facility_id"
        }
    )

    if anomaly_score_column is not None:

        anomalies[
            "raw_anomaly_score"
        ] = pd.to_numeric(
            anomalies[
                anomaly_score_column
            ],
            errors="coerce"
        ).fillna(0.0)

        # Because anomaly scores can contain very large values,
        # normalize them rather than interpreting them as
        # probabilities.

        anomalies[
            "anomaly_evidence"
        ] = normalize_series(
            anomalies[
                "raw_anomaly_score"
            ]
        )

    else:

        anomalies[
            "anomaly_evidence"
        ] = 0.0

    # Strengthen evidence for shortage-specific anomalies.

    shortage_flags = pd.Series(
        0.0,
        index=anomalies.index
    )

    if shortage_anomaly_column is not None:

        shortage_flags += (
            pd.to_numeric(
                anomalies[
                    shortage_anomaly_column
                ],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
        )

    if low_inventory_column is not None:

        shortage_flags += (
            0.5
            *
            pd.to_numeric(
                anomalies[
                    low_inventory_column
                ],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
        )

    if high_consumption_column is not None:

        shortage_flags += (
            0.5
            *
            pd.to_numeric(
                anomalies[
                    high_consumption_column
                ],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
        )

    # Combine normalized anomaly magnitude and
    # shortage-specific evidence.

    anomalies[
        "anomaly_evidence"
    ] = (
        0.50
        * anomalies[
            "anomaly_evidence"
        ]
        +
        0.50
        * shortage_flags.clip(
            0.0,
            1.0
        )
    )

    anomaly_summary = (
        anomalies
        .groupby(
            [
                "facility_id",
                "date"
            ],
            as_index=False
        )
        [
            [
                "anomaly_evidence"
            ]
        ]
        .max()
    )

    # ========================================================
    # 5. LOAD STAGE 8 FEASIBILITY
    # ========================================================

    print("\n" + "=" * 75)
    print("5. LOADING REDISTRIBUTION FEASIBILITY")
    print("=" * 75)

    if not FEASIBILITY_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {FEASIBILITY_FILE}"
        )

    feasibility = pd.read_csv(
        FEASIBILITY_FILE
    )

    print(
        f"Redistribution pairs: "
        f"{len(feasibility):,}"
    )

    print(
        "Feasibility columns:"
    )

    print(
        list(feasibility.columns)
    )

    required_feasibility_columns = [
        "recipient_facility_id",
        "feasibility_score",
    ]

    missing = [
        column
        for column in required_feasibility_columns
        if column not in feasibility.columns
    ]

    if missing:

        raise ValueError(
            "Missing feasibility columns: "
            + ", ".join(missing)
        )

    # For each recipient, keep the strongest available
    # redistribution option.

    feasibility["feasibility_score"] = pd.to_numeric(
        feasibility["feasibility_score"],
        errors="coerce"
    ).fillna(0.0).clip(
        0.0,
        1.0
    )

    best_feasibility = (
        feasibility
        .sort_values(
            "feasibility_score",
            ascending=False
        )
        .groupby(
            "recipient_facility_id",
            as_index=False
        )
        .first()
    )

    best_feasibility = best_feasibility[
        [
            "recipient_facility_id",
            "donor_facility_id",
            "distance_km",
            "donor_surplus_inventory",
            "transit_feasibility_probability",
            "feasibility_score",
            "feasibility_category",
        ]
    ].rename(
        columns={
            "recipient_facility_id":
                "facility_id",

            "feasibility_score":
                "best_feasibility_score",

            "feasibility_category":
                "best_feasibility_category",

            "distance_km":
                "best_donor_distance_km",

            "donor_facility_id":
                "best_donor_facility_id",

            "donor_surplus_inventory":
                "best_donor_surplus",

            "transit_feasibility_probability":
                "best_transit_probability",
        }
    )

    # ========================================================
    # 6. BUILD BASE FACILITY-DATE TABLE
    # ========================================================

    print("\n" + "=" * 75)
    print("6. BUILDING FINAL FACILITY-DATE TABLE")
    print("=" * 75)

    base = predictions[
        [
            "facility_id",
            "date",
            "stockout_risk",
            "stockout",
        ]
    ].copy()

    base = base.rename(
        columns={
            "stockout_risk":
                "ml_shortage_risk",

            "stockout":
                "current_stockout",
        }
    )

    print(
        f"Base records: "
        f"{len(base):,}"
    )

    # ========================================================
    # 7. MERGE SPATIAL SIGNAL
    # ========================================================

    base = base.merge(
        spatial,
        on=[
            "facility_id",
            "date"
        ],
        how="left"
    )

    # ========================================================
    # 8. MERGE ANOMALY SIGNAL
    # ========================================================

    base = base.merge(
        anomaly_summary,
        on=[
            "facility_id",
            "date"
        ],
        how="left"
    )

    # ========================================================
    # 9. FILL MISSING SIGNALS
    # ========================================================

    base[
        "spatial_shortage_pressure"
    ] = base[
        "spatial_shortage_pressure"
    ].fillna(0.0)

    base[
        "spatial_signal"
    ] = base[
        "spatial_signal"
    ].fillna(0).astype(int)

    base[
        "anomaly_evidence"
    ] = base[
        "anomaly_evidence"
    ].fillna(0.0)

    # ========================================================
    # 10. MERGE BEST REDISTRIBUTION OPTION
    # ========================================================

    base = base.merge(
        best_feasibility,
        on="facility_id",
        how="left"
    )

    base[
        "best_feasibility_score"
    ] = base[
        "best_feasibility_score"
    ].fillna(0.0)

    base[
        "best_transit_probability"
    ] = base[
        "best_transit_probability"
    ].fillna(0.0)

    base[
        "best_donor_surplus"
    ] = base[
        "best_donor_surplus"
    ].fillna(0.0)

    base[
        "best_donor_distance_km"
    ] = base[
        "best_donor_distance_km"
    ].fillna(np.nan)

    base[
        "best_feasibility_category"
    ] = base[
        "best_feasibility_category"
    ].fillna("none")

    # ========================================================
    # 11. ADD REGIONAL EVENT INFORMATION
    # ========================================================

    base[
        "regional_event"
    ] = 0

    if not regional.empty:

        regional_date_column = find_column(
            regional,
            [
                "date",
            ],
            required=False
        )

        regional_event_column = find_column(
            regional,
            [
                "regional_event",
            ],
            required=False
        )

        if (
            regional_date_column is not None
            and
            regional_event_column is not None
        ):

            regional_small = regional[
                [
                    regional_date_column,
                    regional_event_column,
                ]
            ].copy()

            regional_small["date"] = pd.to_datetime(
                regional_small[
                    regional_date_column
                ],
                errors="coerce"
            )

            regional_small[
                "regional_event"
            ] = pd.to_numeric(
                regional_small[
                    regional_event_column
                ],
                errors="coerce"
            ).fillna(0).astype(int)

            regional_small = regional_small[
                [
                    "date",
                    "regional_event",
                ]
            ].drop_duplicates(
                "date"
            )

            base = base.drop(
                columns=[
                    "regional_event"
                ]
            )

            base = base.merge(
                regional_small,
                on="date",
                how="left"
            )

            base[
                "regional_event"
            ] = base[
                "regional_event"
            ].fillna(0).astype(int)

    # ========================================================
    # 12. CALCULATE FINAL SCORE
    # ========================================================

    print("\n" + "=" * 75)
    print("7. CALCULATING EVIDENCE-FUSION SCORE")
    print("=" * 75)

    print(
        f"ML weight: "
        f"{ML_WEIGHT:.2f}"
    )

    print(
        f"Spatial weight: "
        f"{SPATIAL_WEIGHT:.2f}"
    )

    print(
        f"Anomaly weight: "
        f"{ANOMALY_WEIGHT:.2f}"
    )

    print(
        f"Feasibility weight: "
        f"{FEASIBILITY_WEIGHT:.2f}"
    )

    weight_sum = (
        ML_WEIGHT
        +
        SPATIAL_WEIGHT
        +
        ANOMALY_WEIGHT
        +
        FEASIBILITY_WEIGHT
    )

    if not np.isclose(
        weight_sum,
        1.0
    ):

        raise ValueError(
            f"Fusion weights must sum to 1. "
            f"Current sum: {weight_sum}"
        )

    # IMPORTANT:
    #
    # Feasibility is evidence that a response option exists.
    # Therefore it contributes positively to the "actionable
    # shortage situation" score.
    #
    # It is NOT saying that redistribution itself causes risk.

    base[
        "final_shortage_score"
    ] = (
        ML_WEIGHT
        * base[
            "ml_shortage_risk"
        ]
        +
        SPATIAL_WEIGHT
        * base[
            "spatial_shortage_pressure"
        ]
        +
        ANOMALY_WEIGHT
        * base[
            "anomaly_evidence"
        ]
        +
        FEASIBILITY_WEIGHT
        * base[
            "best_feasibility_score"
        ]
    )

    base[
        "final_shortage_score"
    ] = base[
        "final_shortage_score"
    ].clip(
        0.0,
        1.0
    )

    # ========================================================
    # 13. RISK LEVEL
    # ========================================================

    base[
        "risk_level"
    ] = base[
        "final_shortage_score"
    ].apply(
        classify_risk
    )

    # ========================================================
    # 14. EVIDENCE FLAGS
    # ========================================================

    base[
        "ml_high_risk_flag"
    ] = (
        base[
            "ml_shortage_risk"
        ]
        >= 0.50
    ).astype(int)

    base[
        "spatial_shortage_flag"
    ] = (
        (
            base[
                "spatial_shortage_pressure"
            ]
            >= 0.50
        )
        |
        (
            base[
                "spatial_signal"
            ]
            == 1
        )
    ).astype(int)

    base[
        "anomaly_flag"
    ] = (
        base[
            "anomaly_evidence"
        ]
        >= 0.50
    ).astype(int)

    base[
        "redistribution_available"
    ] = (
        base[
            "best_feasibility_score"
        ]
        >= 0.50
    ).astype(int)

    # ========================================================
    # 15. EVIDENCE COUNT
    # ========================================================

    base[
        "evidence_count"
    ] = (
        base[
            "ml_high_risk_flag"
        ]
        +
        base[
            "spatial_shortage_flag"
        ]
        +
        base[
            "anomaly_flag"
        ]
        +
        base[
            "current_stockout"
        ]
    )

    # ========================================================
    # 16. HUMAN-READABLE EXPLANATION
    # ========================================================

    base[
        "evidence_summary"
    ] = base.apply(
        build_evidence_text,
        axis=1
    )

    # ========================================================
    # 17. SORT RESULTS
    # ========================================================

    base = base.sort_values(
        [
            "final_shortage_score",
            "ml_shortage_risk",
            "current_stockout",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 18. FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 75)
    print("8. FINAL RESULTS")
    print("=" * 75)

    print(
        f"Facility-date records: "
        f"{len(base):,}"
    )

    print(
        "\nRisk-level distribution:"
    )

    print(
        base[
            "risk_level"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nEvidence-count distribution:"
    )

    print(
        base[
            "evidence_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # 19. TOP ALERTS
    # ========================================================

    print("\n" + "=" * 75)
    print("9. TOP SHORTAGE ALERTS")
    print("=" * 75)

    alert_columns = [
        "date",
        "facility_id",
        "ml_shortage_risk",
        "spatial_shortage_pressure",
        "anomaly_evidence",
        "best_feasibility_score",
        "current_stockout",
        "final_shortage_score",
        "risk_level",
        "evidence_summary",
    ]

    print(
        base[
            alert_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 20. BUILD ALERT TABLE
    # ========================================================

    alerts = base[
        (
            base[
                "final_shortage_score"
            ]
            >= MODERATE_RISK_THRESHOLD
        )
        |
        (
            base[
                "current_stockout"
            ]
            == 1
        )
    ].copy()

    alert_columns = [
        "date",
        "facility_id",
        "ml_shortage_risk",
        "spatial_shortage_pressure",
        "anomaly_evidence",
        "best_feasibility_score",
        "best_donor_facility_id",
        "best_donor_distance_km",
        "best_donor_surplus",
        "best_transit_probability",
        "current_stockout",
        "regional_event",
        "final_shortage_score",
        "risk_level",
        "evidence_count",
        "evidence_summary",
    ]

    # Only retain columns that exist.

    alert_columns = [
        column
        for column in alert_columns
        if column in alerts.columns
    ]

    alerts = alerts[
        alert_columns
    ]

    # ========================================================
    # 21. SAVE OUTPUTS
    # ========================================================

    print("\n" + "=" * 75)
    print("10. SAVING OUTPUTS")
    print("=" * 75)

    FINAL_RISK_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    base.to_csv(
        FINAL_RISK_FILE,
        index=False
    )

    alerts.to_csv(
        FINAL_ALERT_FILE,
        index=False
    )

    print(
        f"Saved: {FINAL_RISK_FILE}"
    )

    print(
        f"Saved: {FINAL_ALERT_FILE}"
    )

    print(
        f"Alert records: "
        f"{len(alerts):,}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 9 COMPLETE")
    print("=" * 75)

    print(
        "\nThe independent shortage signals have been "
        "combined into an explainable facility-level "
        "shortage-risk output."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The final_shortage_score is an evidence-fusion "
        "score, not a calibrated probability."
    )

    print(
        "Redistribution feasibility contains simulated "
        "transport assumptions."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()