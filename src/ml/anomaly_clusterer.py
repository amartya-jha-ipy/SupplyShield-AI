"""
SupplyShield AI
Stage 7 - Demand and Inventory Anomaly Detector

Purpose:
Detect unusual facility-month behaviour for the prototype medicine:
Paracetamol 500 mg.

The detector looks for:
1. Unusually high consumption
2. Unusually low inventory
3. Combined shortage-relevant anomalies

Method:
- Each facility is compared against its own historical behaviour.
- Robust modified Z-scores are used.
- The current observation is never included in its own baseline.
- A minimum history is required before an anomaly is calculated.

Important:
This detector does NOT infer supplier disruption.
Supplier relationships are not present in the real dataset.
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/prototype_paracetamol_500mg.csv"
)

OUTPUT_FILE = Path(
    "data/processed/anomaly_signals.csv"
)

MONTHLY_SUMMARY_FILE = Path(
    "data/processed/monthly_anomaly_summary.csv"
)

# Modified Z-score threshold.
ANOMALY_THRESHOLD = 3.5

# Number of previous observations required
# before calculating an anomaly.
MIN_HISTORY = 6


# ============================================================
# HELPER FUNCTION
# ============================================================

def modified_z_score(value, median_value, mad_value):
    """
    Calculate a robust modified Z-score.

    Formula:

        Modified Z =
            0.6745 * (x - median) / MAD

    If MAD is zero, a fallback relative deviation is used.
    """

    if pd.isna(value) or pd.isna(median_value):
        return np.nan

    if pd.isna(mad_value) or mad_value == 0:

        if median_value == 0:
            return 0.0

        return (
            (value - median_value)
            / abs(median_value)
        )

    return (
        0.6745
        * (value - median_value)
        / mad_value
    )


# ============================================================
# CALCULATE ANOMALIES FOR ONE FACILITY
# ============================================================

def calculate_facility_anomalies(group):
    """
    Calculate historical anomaly scores for one facility.

    Important:
    The current month is scored against previous observations only.
    """

    group = group.sort_values("date").copy()

    consumption_history = []
    inventory_history = []

    consumption_scores = []
    inventory_scores = []

    consumption_medians = []
    consumption_mads = []

    inventory_medians = []
    inventory_mads = []

    for _, row in group.iterrows():

        current_consumption = row["consumption"]
        current_inventory = row["closing_inventory"]

        # ----------------------------------------------------
        # Consumption baseline
        # ----------------------------------------------------

        if len(consumption_history) >= MIN_HISTORY:

            consumption_array = np.array(
                consumption_history,
                dtype=float
            )

            consumption_median = float(
                np.median(consumption_array)
            )

            consumption_mad = float(
                np.median(
                    np.abs(
                        consumption_array
                        - consumption_median
                    )
                )
            )

            consumption_score = modified_z_score(
                current_consumption,
                consumption_median,
                consumption_mad
            )

        else:

            consumption_median = np.nan
            consumption_mad = np.nan
            consumption_score = np.nan

        # ----------------------------------------------------
        # Inventory baseline
        # ----------------------------------------------------

        if len(inventory_history) >= MIN_HISTORY:

            inventory_array = np.array(
                inventory_history,
                dtype=float
            )

            inventory_median = float(
                np.median(inventory_array)
            )

            inventory_mad = float(
                np.median(
                    np.abs(
                        inventory_array
                        - inventory_median
                    )
                )
            )

            inventory_score = modified_z_score(
                current_inventory,
                inventory_median,
                inventory_mad
            )

        else:

            inventory_median = np.nan
            inventory_mad = np.nan
            inventory_score = np.nan

        # ----------------------------------------------------
        # Save scores
        # ----------------------------------------------------

        consumption_scores.append(
            consumption_score
        )

        inventory_scores.append(
            inventory_score
        )

        consumption_medians.append(
            consumption_median
        )

        consumption_mads.append(
            consumption_mad
        )

        inventory_medians.append(
            inventory_median
        )

        inventory_mads.append(
            inventory_mad
        )

        # ----------------------------------------------------
        # Add current observation AFTER scoring.
        #
        # This prevents the current observation from
        # influencing its own historical baseline.
        # ----------------------------------------------------

        if not pd.isna(current_consumption):
            consumption_history.append(
                current_consumption
            )

        if not pd.isna(current_inventory):
            inventory_history.append(
                current_inventory
            )

    group["consumption_modified_z"] = (
        consumption_scores
    )

    group["inventory_modified_z"] = (
        inventory_scores
    )

    group["historical_consumption_median"] = (
        consumption_medians
    )

    group["historical_consumption_mad"] = (
        consumption_mads
    )

    group["historical_inventory_median"] = (
        inventory_medians
    )

    group["historical_inventory_mad"] = (
        inventory_mads
    )

    return group


# ============================================================
# ANOMALY SEVERITY
# ============================================================

def classify_severity(score):
    """
    Convert combined anomaly score into a simple severity label.
    """

    if pd.isna(score):
        return "insufficient_history"

    if score >= 7:
        return "extreme"

    if score >= 5:
        return "high"

    if score >= ANOMALY_THRESHOLD:
        return "moderate"

    return "normal"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - DEMAND / INVENTORY ANOMALY DETECTOR")
    print("=" * 75)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n" + "=" * 75)
    print("1. LOADING PROTOTYPE DATA")
    print("=" * 75)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded records: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # ========================================================
    # 2. VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "facility_id",
        "medicine",
        "date",
        "consumption",
        "closing_inventory",
        "stockout",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ========================================================
    # 3. PREPARE DATA
    # ========================================================

    print("\n" + "=" * 75)
    print("2. PREPARING DATA")
    print("=" * 75)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["consumption"] = pd.to_numeric(
        df["consumption"],
        errors="coerce"
    )

    df["closing_inventory"] = pd.to_numeric(
        df["closing_inventory"],
        errors="coerce"
    )

    df["stockout"] = pd.to_numeric(
        df["stockout"],
        errors="coerce"
    )

    rows_before = len(df)

    df = df.dropna(
        subset=[
            "facility_id",
            "date",
            "consumption",
            "closing_inventory",
        ]
    ).copy()

    rows_removed = (
        rows_before
        - len(df)
    )

    print(
        f"Rows removed because of missing values: "
        f"{rows_removed:,}"
    )

    print(
        f"Rows remaining: {len(df):,}"
    )

    df = df.sort_values(
        [
            "facility_id",
            "date"
        ]
    ).reset_index(drop=True)

    # ========================================================
    # 4. BASIC INFORMATION
    # ========================================================

    print("\n" + "=" * 75)
    print("3. DATASET INFORMATION")
    print("=" * 75)

    facility_count = (
        df["facility_id"].nunique()
    )

    date_count = (
        df["date"].nunique()
    )

    print(
        f"Facilities: {facility_count:,}"
    )

    print(
        f"Dates: {date_count:,}"
    )

    print(
        f"Minimum historical observations: "
        f"{MIN_HISTORY}"
    )

    # ========================================================
    # 5. CALCULATE FACILITY ANOMALIES
    # ========================================================

    print("\n" + "=" * 75)
    print("4. CALCULATING FACILITY HISTORICAL ANOMALIES")
    print("=" * 75)

    processed_groups = []

    total_facilities = (
        df["facility_id"].nunique()
    )

    for counter, (
        facility_id,
        group
    ) in enumerate(
        df.groupby("facility_id"),
        start=1
    ):

        processed_group = (
            calculate_facility_anomalies(group)
        )

        processed_groups.append(
            processed_group
        )

        if counter % 200 == 0:

            print(
                f"Processed facilities: "
                f"{counter:,} / "
                f"{total_facilities:,}"
            )

    df = pd.concat(
        processed_groups,
        ignore_index=True
    )

    df = df.sort_values(
        [
            "facility_id",
            "date"
        ]
    ).reset_index(drop=True)

    print(
        f"Processed facilities: "
        f"{total_facilities:,} / "
        f"{total_facilities:,}"
    )

    # ========================================================
    # 6. ABSOLUTE ANOMALY MAGNITUDES
    # ========================================================

    print("\n" + "=" * 75)
    print("5. CALCULATING ANOMALY MAGNITUDES")
    print("=" * 75)

    df["abs_consumption_anomaly"] = (
        df["consumption_modified_z"].abs()
    )

    df["abs_inventory_anomaly"] = (
        df["inventory_modified_z"].abs()
    )

    # ========================================================
    # 7. COMBINED ANOMALY SCORE
    # ========================================================

    print(
        "Combining consumption and inventory anomaly evidence..."
    )

    df["combined_anomaly_score"] = (
        0.5
        * df["abs_consumption_anomaly"].fillna(0)
        +
        0.5
        * df["abs_inventory_anomaly"].fillna(0)
    )

    # ========================================================
    # 8. GENERAL ANOMALY FLAGS
    # ========================================================

    df["consumption_anomaly"] = (
        df["abs_consumption_anomaly"]
        >= ANOMALY_THRESHOLD
    )

    df["inventory_anomaly"] = (
        df["abs_inventory_anomaly"]
        >= ANOMALY_THRESHOLD
    )

    df["combined_anomaly"] = (
        df["consumption_anomaly"]
        |
        df["inventory_anomaly"]
    )

    # ========================================================
    # 9. SHORTAGE-RELEVANT ANOMALIES
    # ========================================================

    # Negative inventory Z-score:
    # inventory is unusually LOW.

    df["low_inventory_anomaly"] = (
        df["inventory_modified_z"]
        <= -ANOMALY_THRESHOLD
    )

    # Positive consumption Z-score:
    # consumption is unusually HIGH.

    df["high_consumption_anomaly"] = (
        df["consumption_modified_z"]
        >= ANOMALY_THRESHOLD
    )

    # Either unusually low inventory OR unusually high demand
    # is relevant to shortage propagation.

    df["shortage_anomaly"] = (
        df["low_inventory_anomaly"]
        |
        df["high_consumption_anomaly"]
    )

    # ========================================================
    # 10. SEVERITY
    # ========================================================

    df["anomaly_severity"] = (
        df["combined_anomaly_score"]
        .apply(classify_severity)
    )

    # ========================================================
    # 11. MONTHLY SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("6. BUILDING MONTHLY ANOMALY SUMMARY")
    print("=" * 75)

    monthly_summary = (
        df.groupby("date")
        .agg(
            observed_facilities=(
                "facility_id",
                "nunique"
            ),
            consumption_anomalies=(
                "consumption_anomaly",
                "sum"
            ),
            inventory_anomalies=(
                "inventory_anomaly",
                "sum"
            ),
            shortage_anomalies=(
                "shortage_anomaly",
                "sum"
            ),
            high_consumption_facilities=(
                "high_consumption_anomaly",
                "sum"
            ),
            low_inventory_facilities=(
                "low_inventory_anomaly",
                "sum"
            ),
            stockout_facilities=(
                "stockout",
                "sum"
            ),
            average_consumption=(
                "consumption",
                "mean"
            ),
            average_inventory=(
                "closing_inventory",
                "mean"
            ),
            maximum_anomaly_score=(
                "combined_anomaly_score",
                "max"
            ),
        )
        .reset_index()
    )

    monthly_summary[
        "shortage_anomaly_fraction"
    ] = (
        monthly_summary[
            "shortage_anomalies"
        ]
        /
        monthly_summary[
            "observed_facilities"
        ]
    )

    # ========================================================
    # 12. RESULTS
    # ========================================================

    print("\n" + "=" * 75)
    print("7. RESULTS")
    print("=" * 75)

    usable_rows = df[
        df["consumption_modified_z"].notna()
        |
        df["inventory_modified_z"].notna()
    ]

    print(
        f"Facility-month signals: "
        f"{len(df):,}"
    )

    print(
        f"Rows with sufficient history: "
        f"{len(usable_rows):,}"
    )

    print(
        f"Consumption anomalies: "
        f"{int(df['consumption_anomaly'].sum()):,}"
    )

    print(
        f"Inventory anomalies: "
        f"{int(df['inventory_anomaly'].sum()):,}"
    )

    print(
        f"Shortage-relevant anomalies: "
        f"{int(df['shortage_anomaly'].sum()):,}"
    )

    print(
        f"Low-inventory anomalies: "
        f"{int(df['low_inventory_anomaly'].sum()):,}"
    )

    print(
        f"High-consumption anomalies: "
        f"{int(df['high_consumption_anomaly'].sum()):,}"
    )

    # ========================================================
    # 13. TOP SHORTAGE ANOMALIES
    # ========================================================

    print("\n" + "=" * 75)
    print("8. STRONGEST SHORTAGE-RELEVANT ANOMALIES")
    print("=" * 75)

    top_anomalies = (
        df[
            df["shortage_anomaly"]
        ]
        .sort_values(
            "combined_anomaly_score",
            ascending=False
        )
        [
            [
                "date",
                "facility_id",
                "district",
                "consumption",
                "closing_inventory",
                "consumption_modified_z",
                "inventory_modified_z",
                "combined_anomaly_score",
                "stockout",
            ]
        ]
        .head(20)
    )

    if len(top_anomalies) == 0:

        print(
            "No shortage-relevant anomalies detected."
        )

    else:

        print(
            top_anomalies.to_string(
                index=False
            )
        )

    # ========================================================
    # 14. MONTHLY SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("9. MONTHLY SUMMARY")
    print("=" * 75)

    display_summary = monthly_summary[
        [
            "date",
            "observed_facilities",
            "shortage_anomalies",
            "low_inventory_facilities",
            "high_consumption_facilities",
            "stockout_facilities",
            "shortage_anomaly_fraction",
        ]
    ].copy()

    display_summary[
        "shortage_anomaly_fraction"
    ] = (
        display_summary[
            "shortage_anomaly_fraction"
        ]
        * 100
    ).round(2)

    print(
        display_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 15. SAVE OUTPUTS
    # ========================================================

    print("\n" + "=" * 75)
    print("10. SAVING OUTPUTS")
    print("=" * 75)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_columns = [
        "facility_id",
        "medicine",
        "date",
        "district",
        "facility_type",
        "latitude",
        "longitude",
        "consumption",
        "closing_inventory",
        "stockout",
        "consumption_modified_z",
        "inventory_modified_z",
        "combined_anomaly_score",
        "consumption_anomaly",
        "inventory_anomaly",
        "low_inventory_anomaly",
        "high_consumption_anomaly",
        "shortage_anomaly",
        "anomaly_severity",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    df[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False
    )

    monthly_summary.to_csv(
        MONTHLY_SUMMARY_FILE,
        index=False
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Saved: {MONTHLY_SUMMARY_FILE}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 7 COMPLETE")
    print("=" * 75)

    print(
        "\nAnomaly signals are now ready to be combined "
        "with the ML shortage-risk and spatial signals."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()