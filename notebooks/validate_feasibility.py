import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FEASIBILITY_FILE = (
    "data/processed/redistribution_feasibility.csv"
)

OUTPUT_DIR = "data/processed"

VALIDATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "feasibility_validation.csv"
)

CATEGORY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "feasibility_category_validation.csv"
)

RECIPIENT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "feasibility_recipient_validation.csv"
)


# ============================================================
# EXACT COLUMNS FROM ACTUAL STAGE 8 OUTPUT
# ============================================================

REQUIRED_COLUMNS = [
    "donor_facility_id",
    "recipient_facility_id",
    "donor_district",
    "recipient_district",
    "distance_km",
    "donor_inventory",
    "donor_surplus_inventory",
    "recipient_inventory",
    "recipient_monthly_demand",
    "recipient_stockout_probability",
    "recipient_safety_adjusted_demand",
    "recipient_stockout_time_months",
    "transit_feasibility_probability",
    "distance_factor",
    "surplus_factor",
    "recipient_urgency",
    "feasibility_score",
    "feasibility_category",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - REDISTRIBUTION FEASIBILITY VALIDATION")
    print("=" * 75)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n" + "=" * 75)
    print("1. LOADING REDISTRIBUTION FEASIBILITY OUTPUT")
    print("=" * 75)

    if not os.path.exists(FEASIBILITY_FILE):
        raise FileNotFoundError(
            f"Feasibility file not found: {FEASIBILITY_FILE}"
        )

    df = pd.read_csv(FEASIBILITY_FILE)

    print(
        f"Feasibility records: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # ========================================================
    # 2. EXACT COLUMN VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("2. VALIDATING COLUMN NAMES")
    print("=" * 75)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "ERROR: Missing required columns:"
        )

        for column in missing_columns:
            print(
                f"  - {column}"
            )

        print(
            "\nActual columns found:"
        )

        for column in df.columns:
            print(
                f"  - {column}"
            )

        raise ValueError(
            "Column-name mismatch detected. "
            "No validation was performed."
        )

    print(
        "All required Stage 8 column names found."
    )

    print(
        "Column-name check: PASS"
    )

    # ========================================================
    # 3. NUMERIC VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("3. VALIDATING NUMERIC VALUES")
    print("=" * 75)

    numeric_columns = [
        "distance_km",
        "donor_inventory",
        "donor_surplus_inventory",
        "recipient_inventory",
        "recipient_monthly_demand",
        "recipient_stockout_probability",
        "recipient_safety_adjusted_demand",
        "recipient_stockout_time_months",
        "transit_feasibility_probability",
        "distance_factor",
        "surplus_factor",
        "recipient_urgency",
        "feasibility_score",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    invalid_numeric_rows = int(
        df[numeric_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"Rows with invalid numeric values: "
        f"{invalid_numeric_rows:,}"
    )

    # ========================================================
    # 4. BASIC DATASET STATISTICS
    # ========================================================

    print("\n" + "=" * 75)
    print("4. BASIC REDISTRIBUTION STATISTICS")
    print("=" * 75)

    unique_donors = df[
        "donor_facility_id"
    ].nunique()

    unique_recipients = df[
        "recipient_facility_id"
    ].nunique()

    print(
        f"Unique donor facilities: "
        f"{unique_donors:,}"
    )

    print(
        f"Unique recipient facilities: "
        f"{unique_recipients:,}"
    )

    print(
        f"Average donor surplus: "
        f"{df['donor_surplus_inventory'].mean():.2f}"
    )

    print(
        f"Median donor surplus: "
        f"{df['donor_surplus_inventory'].median():.2f}"
    )

    print(
        f"Average recipient inventory: "
        f"{df['recipient_inventory'].mean():.2f}"
    )

    print(
        f"Average recipient monthly demand: "
        f"{df['recipient_monthly_demand'].mean():.2f}"
    )

    # ========================================================
    # 5. RANGE VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("5. RANGE VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    negative_distance = int(
        (
            df["distance_km"] < 0
        ).sum()
    )

    print(
        f"Distance range: "
        f"{df['distance_km'].min():.2f} - "
        f"{df['distance_km'].max():.2f} km"
    )

    print(
        f"Negative distances: "
        f"{negative_distance:,}"
    )

    # --------------------------------------------------------
    # Donor inventory
    # --------------------------------------------------------

    negative_donor_inventory = int(
        (
            df["donor_inventory"] < 0
        ).sum()
    )

    print(
        f"\nDonor inventory range: "
        f"{df['donor_inventory'].min():.2f} - "
        f"{df['donor_inventory'].max():.2f}"
    )

    print(
        f"Negative donor inventory values: "
        f"{negative_donor_inventory:,}"
    )

    # --------------------------------------------------------
    # Donor surplus
    # --------------------------------------------------------

    negative_surplus = int(
        (
            df["donor_surplus_inventory"] < 0
        ).sum()
    )

    print(
        f"\nDonor surplus range: "
        f"{df['donor_surplus_inventory'].min():.2f} - "
        f"{df['donor_surplus_inventory'].max():.2f}"
    )

    print(
        f"Negative donor surplus values: "
        f"{negative_surplus:,}"
    )

    # --------------------------------------------------------
    # Recipient inventory
    # --------------------------------------------------------

    negative_inventory = int(
        (
            df["recipient_inventory"] < 0
        ).sum()
    )

    print(
        f"\nRecipient inventory range: "
        f"{df['recipient_inventory'].min():.2f} - "
        f"{df['recipient_inventory'].max():.2f}"
    )

    print(
        f"Negative recipient inventory values: "
        f"{negative_inventory:,}"
    )

    # --------------------------------------------------------
    # Recipient demand
    # --------------------------------------------------------

    negative_demand = int(
        (
            df["recipient_monthly_demand"] < 0
        ).sum()
    )

    print(
        f"\nRecipient monthly demand range: "
        f"{df['recipient_monthly_demand'].min():.2f} - "
        f"{df['recipient_monthly_demand'].max():.2f}"
    )

    print(
        f"Negative recipient demand values: "
        f"{negative_demand:,}"
    )

    # ========================================================
    # 6. PROBABILITY VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("6. PROBABILITY VALIDATION")
    print("=" * 75)

    probability_columns = [
        "recipient_stockout_probability",
        "transit_feasibility_probability",
        "feasibility_score",
    ]

    probability_results = {}

    for column in probability_columns:

        values = df[column]

        invalid = int(
            (
                (values < 0)
                |
                (values > 1)
            ).sum()
        )

        probability_results[column] = invalid

        print(
            f"\n{column}:"
        )

        print(
            f"  range = "
            f"{values.min():.4f} - "
            f"{values.max():.4f}"
        )

        print(
            f"  outside [0,1] = "
            f"{invalid:,}"
        )

    # ========================================================
    # 7. FACTOR VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("7. FEASIBILITY FACTOR VALIDATION")
    print("=" * 75)

    factor_columns = [
        "distance_factor",
        "surplus_factor",
        "recipient_urgency",
    ]

    for column in factor_columns:

        values = df[column]

        invalid = int(
            (
                (values < 0)
                |
                (values > 1)
            ).sum()
        )

        print(
            f"\n{column}:"
        )

        print(
            f"  range = "
            f"{values.min():.4f} - "
            f"{values.max():.4f}"
        )

        print(
            f"  outside [0,1] = "
            f"{invalid:,}"
        )

    # ========================================================
    # 8. STOCKOUT TIME VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("8. STOCKOUT-TIME VALIDATION")
    print("=" * 75)

    stockout_time = df[
        "recipient_stockout_time_months"
    ]

    print(
        f"Stockout-time range: "
        f"{stockout_time.min():.4f} - "
        f"{stockout_time.max():.4f} months"
    )

    negative_stockout_time = int(
        (
            stockout_time < 0
        ).sum()
    )

    print(
        f"Negative stockout times: "
        f"{negative_stockout_time:,}"
    )

    zero_stockout_time = int(
        (
            stockout_time == 0
        ).sum()
    )

    print(
        f"Zero stockout-time records: "
        f"{zero_stockout_time:,}"
    )

    # ========================================================
    # 9. DONOR / RECIPIENT LOGIC
    # ========================================================

    print("\n" + "=" * 75)
    print("9. DONOR / RECIPIENT LOGIC VALIDATION")
    print("=" * 75)

    same_facility = int(
        (
            df["donor_facility_id"]
            ==
            df["recipient_facility_id"]
        ).sum()
    )

    print(
        f"Donor = recipient pairs: "
        f"{same_facility:,}"
    )

    positive_surplus = int(
        (
            df["donor_surplus_inventory"]
            > 0
        ).sum()
    )

    zero_surplus = int(
        (
            df["donor_surplus_inventory"]
            == 0
        ).sum()
    )

    print(
        f"Pairs with positive donor surplus: "
        f"{positive_surplus:,}"
    )

    print(
        f"Pairs with zero donor surplus: "
        f"{zero_surplus:,}"
    )

    # ========================================================
    # 10. RECIPIENT URGENCY ANALYSIS
    # ========================================================

    print("\n" + "=" * 75)
    print("10. RECIPIENT URGENCY ANALYSIS")
    print("=" * 75)

    high_risk = int(
        (
            df["recipient_stockout_probability"]
            >= 0.70
        ).sum()
    )

    moderate_high_risk = int(
        (
            df["recipient_stockout_probability"]
            >= 0.50
        ).sum()
    )

    low_inventory = int(
        (
            df["recipient_inventory"]
            <=
            df["recipient_monthly_demand"]
        ).sum()
    )

    urgent_stockout = int(
        (
            df["recipient_stockout_time_months"]
            <= 1
        ).sum()
    )

    print(
        f"Pairs with shortage probability >= 0.50: "
        f"{moderate_high_risk:,}"
    )

    print(
        f"Pairs with shortage probability >= 0.70: "
        f"{high_risk:,}"
    )

    print(
        f"Pairs where inventory <= monthly demand: "
        f"{low_inventory:,}"
    )

    print(
        f"Pairs with stockout time <= 1 month: "
        f"{urgent_stockout:,}"
    )

    # ========================================================
    # 11. FEASIBILITY CATEGORY ANALYSIS
    # ========================================================

    print("\n" + "=" * 75)
    print("11. FEASIBILITY CATEGORY ANALYSIS")
    print("=" * 75)

    category_counts = (
        df["feasibility_category"]
        .value_counts(dropna=False)
    )

    print(
        category_counts.to_string()
    )

    category_summary = (
        df
        .groupby(
            "feasibility_category",
            dropna=False
        )
        .agg(
            pairs=(
                "feasibility_score",
                "count"
            ),
            average_distance_km=(
                "distance_km",
                "mean"
            ),
            average_shortage_probability=(
                "recipient_stockout_probability",
                "mean"
            ),
            average_transit_probability=(
                "transit_feasibility_probability",
                "mean"
            ),
            average_feasibility_score=(
                "feasibility_score",
                "mean"
            ),
            average_donor_surplus=(
                "donor_surplus_inventory",
                "mean"
            ),
            average_stockout_time_months=(
                "recipient_stockout_time_months",
                "mean"
            ),
        )
        .reset_index()
        .sort_values(
            "average_feasibility_score",
            ascending=False
        )
    )

    print(
        "\nCategory summary:"
    )

    print(
        category_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 12. DISTANCE VS FEASIBILITY
    # ========================================================

    print("\n" + "=" * 75)
    print("12. DISTANCE / FEASIBILITY ANALYSIS")
    print("=" * 75)

    if len(df) >= 2:

        distance_corr = (
            df[
                [
                    "distance_km",
                    "feasibility_score"
                ]
            ]
            .corr()
            .iloc[0, 1]
        )

        print(
            f"Correlation between distance and "
            f"feasibility score: "
            f"{distance_corr:.4f}"
        )

    df["distance_band"] = pd.cut(
        df["distance_km"],
        bins=[
            -np.inf,
            10,
            25,
            50,
            100,
            np.inf,
        ],
        labels=[
            "<=10 km",
            "10-25 km",
            "25-50 km",
            "50-100 km",
            ">100 km",
        ],
    )

    distance_summary = (
        df
        .groupby(
            "distance_band",
            observed=False
        )
        .agg(
            pairs=(
                "feasibility_score",
                "count"
            ),
            average_feasibility=(
                "feasibility_score",
                "mean"
            ),
            average_transit_probability=(
                "transit_feasibility_probability",
                "mean"
            ),
            average_shortage_probability=(
                "recipient_stockout_probability",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        "\nDistance-band summary:"
    )

    print(
        distance_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 13. HIGH-VALUE REDISTRIBUTION OPTIONS
    # ========================================================

    print("\n" + "=" * 75)
    print("13. HIGH-VALUE REDISTRIBUTION OPTIONS")
    print("=" * 75)

    top_options = (
        df[
            [
                "donor_facility_id",
                "recipient_facility_id",
                "distance_km",
                "donor_surplus_inventory",
                "recipient_inventory",
                "recipient_monthly_demand",
                "recipient_stockout_probability",
                "recipient_stockout_time_months",
                "transit_feasibility_probability",
                "feasibility_score",
                "feasibility_category",
            ]
        ]
        .sort_values(
            [
                "feasibility_score",
                "recipient_stockout_probability",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(10)
    )

    print(
        top_options.to_string(
            index=False
        )
    )

    # ========================================================
    # 14. RECIPIENT-LEVEL SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("14. RECIPIENT-LEVEL VALIDATION")
    print("=" * 75)

    recipient_summary = (
        df
        .groupby(
            "recipient_facility_id"
        )
        .agg(
            available_donor_options=(
                "donor_facility_id",
                "count"
            ),
            best_feasibility_score=(
                "feasibility_score",
                "max"
            ),
            best_transit_probability=(
                "transit_feasibility_probability",
                "max"
            ),
            minimum_distance_km=(
                "distance_km",
                "min"
            ),
            shortage_probability=(
                "recipient_stockout_probability",
                "max"
            ),
            stockout_time_months=(
                "recipient_stockout_time_months",
                "min"
            ),
        )
        .reset_index()
        .sort_values(
            "best_feasibility_score",
            ascending=False
        )
    )

    print(
        f"Recipients with at least one donor option: "
        f"{len(recipient_summary):,}"
    )

    print(
        "\nTop recipient summaries:"
    )

    print(
        recipient_summary
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # 15. SCORE SANITY CHECK
    # ========================================================

    print("\n" + "=" * 75)
    print("15. INTERNAL SCORE SANITY CHECK")
    print("=" * 75)

    score_valid = bool(
        (
            (df["feasibility_score"] >= 0)
            &
            (df["feasibility_score"] <= 1)
        ).all()
    )

    transit_valid = bool(
        (
            (
                df[
                    "transit_feasibility_probability"
                ]
                >= 0
            )
            &
            (
                df[
                    "transit_feasibility_probability"
                ]
                <= 1
            )
        ).all()
    )

    shortage_probability_valid = bool(
        (
            (
                df[
                    "recipient_stockout_probability"
                ]
                >= 0
            )
            &
            (
                df[
                    "recipient_stockout_probability"
                ]
                <= 1
            )
        ).all()
    )

    distance_factor_valid = bool(
        (
            (df["distance_factor"] >= 0)
            &
            (df["distance_factor"] <= 1)
        ).all()
    )

    surplus_factor_valid = bool(
        (
            (df["surplus_factor"] >= 0)
            &
            (df["surplus_factor"] <= 1)
        ).all()
    )

    urgency_valid = bool(
        (
            (df["recipient_urgency"] >= 0)
            &
            (df["recipient_urgency"] <= 1)
        ).all()
    )

    print(
        f"Feasibility score in [0,1]: "
        f"{'YES' if score_valid else 'NO'}"
    )

    print(
        f"Transit probability in [0,1]: "
        f"{'YES' if transit_valid else 'NO'}"
    )

    print(
        f"Shortage probability in [0,1]: "
        f"{'YES' if shortage_probability_valid else 'NO'}"
    )

    print(
        f"Distance factor in [0,1]: "
        f"{'YES' if distance_factor_valid else 'NO'}"
    )

    print(
        f"Surplus factor in [0,1]: "
        f"{'YES' if surplus_factor_valid else 'NO'}"
    )

    print(
        f"Recipient urgency in [0,1]: "
        f"{'YES' if urgency_valid else 'NO'}"
    )

    # ========================================================
    # 16. SAVE VALIDATION OUTPUTS
    # ========================================================

    print("\n" + "=" * 75)
    print("16. SAVING VALIDATION OUTPUTS")
    print("=" * 75)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "feasibility_records",
                "value": len(df),
            },
            {
                "metric": "unique_donors",
                "value": unique_donors,
            },
            {
                "metric": "unique_recipients",
                "value": unique_recipients,
            },
            {
                "metric": "invalid_numeric_rows",
                "value": invalid_numeric_rows,
            },
            {
                "metric": "negative_distance_rows",
                "value": negative_distance,
            },
            {
                "metric": "negative_donor_inventory_rows",
                "value": negative_donor_inventory,
            },
            {
                "metric": "negative_donor_surplus_rows",
                "value": negative_surplus,
            },
            {
                "metric": "negative_recipient_inventory_rows",
                "value": negative_inventory,
            },
            {
                "metric": "negative_recipient_demand_rows",
                "value": negative_demand,
            },
            {
                "metric": "negative_stockout_time_rows",
                "value": negative_stockout_time,
            },
            {
                "metric": "same_facility_pairs",
                "value": same_facility,
            },
            {
                "metric": "positive_surplus_pairs",
                "value": positive_surplus,
            },
            {
                "metric": "high_risk_pairs",
                "value": high_risk,
            },
            {
                "metric": "urgent_stockout_time_le_1_month",
                "value": urgent_stockout,
            },
            {
                "metric": "feasibility_score_valid",
                "value": score_valid,
            },
            {
                "metric": "transit_probability_valid",
                "value": transit_valid,
            },
            {
                "metric": "shortage_probability_valid",
                "value": shortage_probability_valid,
            },
            {
                "metric": "distance_factor_valid",
                "value": distance_factor_valid,
            },
            {
                "metric": "surplus_factor_valid",
                "value": surplus_factor_valid,
            },
            {
                "metric": "recipient_urgency_valid",
                "value": urgency_valid,
            },
        ]
    )

    summary.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    print(
        f"Saved: {VALIDATION_OUTPUT}"
    )

    category_summary.to_csv(
        CATEGORY_OUTPUT,
        index=False
    )

    print(
        f"Saved: {CATEGORY_OUTPUT}"
    )

    recipient_summary.to_csv(
        RECIPIENT_OUTPUT,
        index=False
    )

    print(
        f"Saved: {RECIPIENT_OUTPUT}"
    )

    # ========================================================
    # 17. FINAL STATUS
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 11F COMPLETE")
    print("=" * 75)

    print(
        "Redistribution feasibility outputs were "
        "validated for exact column consistency, "
        "numeric ranges, donor/recipient logic, "
        "recipient urgency, geographic distance, "
        "and feasibility behavior."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Transit times, donor-recipient relationships, "
        "and transport feasibility are simulated "
        "prototype assumptions."
    )

    print(
        "They are not observed supplier or transport "
        "measurements from the real dataset."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()