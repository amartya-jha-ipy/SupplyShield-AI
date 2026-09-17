"""
SupplyShield AI
Stage 8 - Redistribution Feasibility Scorer

Purpose:
Estimate whether medicine can be transferred from a facility with
surplus inventory to a facility facing shortage risk.

Real data:
- Facility locations
- Current inventory
- Consumption
- Facility identity
- ML shortage-risk predictions

Simulated assumptions:
- Transit time
- Safety margin
- Donor/recipient relationships

IMPORTANT:
Transit times and redistribution relationships are simulated
prototype assumptions. They are NOT observed data from the
real healthcare dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROTOTYPE_FILE = Path(
    "data/processed/prototype_paracetamol_500mg.csv"
)

PREDICTION_FILE = Path(
    "data/processed/shortage_risk_predictions.csv"
)

OUTPUT_FILE = Path(
    "data/processed/redistribution_feasibility.csv"
)

TARGET_SUMMARY_FILE = Path(
    "data/processed/redistribution_targets.csv"
)

# Monte Carlo simulations per donor-recipient pair.
N_SIMULATIONS = 1000

RANDOM_SEED = 42

# Safety margin on expected demand.
# 0.20 = 20% additional demand buffer.
SAFETY_MARGIN = 0.20

# Minimum inventory coverage for a potential donor.
DONOR_COVERAGE_THRESHOLD = 2.0

# Minimum predicted stockout risk for a recipient.
RECIPIENT_RISK_THRESHOLD = 0.50

# Maximum geographic distance considered.
MAX_DISTANCE_KM = 100.0

# Simulated median transit time in months.
#
# This is an assumption for the prototype.
TRANSIT_MEDIAN_MONTHS = 0.50

# LogNormal spread.
TRANSIT_SIGMA = 0.35

# Donor retains two months of safety-adjusted demand.
DONOR_SAFETY_MONTHS = 2.0


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate geographic distance between two coordinates.

    Returns kilometres.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(
        np.sqrt(a)
    )

    earth_radius_km = 6371.0

    return earth_radius_km * c


# ============================================================
# MONTE CARLO TRANSIT PROBABILITY
# ============================================================

def calculate_transit_probability(
    stockout_time_months,
    rng
):
    """
    Estimate:

        P(transit time <= stockout time)

    using Monte Carlo simulation.

    Transit time follows a LogNormal distribution.

    Returns a value between 0 and 1.
    """

    if pd.isna(stockout_time_months):
        return np.nan

    if stockout_time_months <= 0:
        return 0.0

    log_mean = np.log(
        TRANSIT_MEDIAN_MONTHS
    )

    simulated_transit = rng.lognormal(
        mean=log_mean,
        sigma=TRANSIT_SIGMA,
        size=N_SIMULATIONS
    )

    probability = np.mean(
        simulated_transit
        <= stockout_time_months
    )

    return float(probability)


# ============================================================
# FEASIBILITY CATEGORY
# ============================================================

def classify_feasibility(score):
    """
    Convert feasibility score into a simple category.
    """

    if pd.isna(score):
        return "unknown"

    if score >= 0.75:
        return "high"

    if score >= 0.50:
        return "moderate"

    if score >= 0.25:
        return "low"

    return "very_low"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SUPPLYSHIELD AI - REDISTRIBUTION FEASIBILITY SCORER")
    print("=" * 75)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # ========================================================
    # 1. LOAD PROTOTYPE DATA
    # ========================================================

    print("\n" + "=" * 75)
    print("1. LOADING PROTOTYPE DATA")
    print("=" * 75)

    if not PROTOTYPE_FILE.exists():

        raise FileNotFoundError(
            f"Prototype file not found: {PROTOTYPE_FILE}"
        )

    df = pd.read_csv(
        PROTOTYPE_FILE
    )

    print(
        f"Prototype records: {len(df):,}"
    )

    # ========================================================
    # 2. LOAD ML PREDICTIONS
    # ========================================================

    print("\n" + "=" * 75)
    print("2. LOADING SHORTAGE-RISK PREDICTIONS")
    print("=" * 75)

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    print(
        f"Prediction records: {len(predictions):,}"
    )

    print(
        "Prediction columns:"
    )

    print(
        list(predictions.columns)
    )

    # ========================================================
    # 3. VALIDATE PREDICTION COLUMNS
    # ========================================================

    print("\n" + "=" * 75)
    print("3. VALIDATING PREDICTION DATA")
    print("=" * 75)

    required_prediction_columns = [
        "facility_id",
        "date",
        "stockout_risk",
    ]

    missing_prediction_columns = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing_prediction_columns:

        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(
                missing_prediction_columns
            )
        )

    print(
        "Using ML prediction column: stockout_risk"
    )

    # ========================================================
    # 4. VALIDATE PROTOTYPE COLUMNS
    # ========================================================

    required_prototype_columns = [
        "facility_id",
        "date",
        "closing_inventory",
        "consumption",
        "latitude",
        "longitude",
    ]

    missing_prototype_columns = [
        column
        for column in required_prototype_columns
        if column not in df.columns
    ]

    if missing_prototype_columns:

        raise ValueError(
            "Missing prototype columns: "
            + ", ".join(
                missing_prototype_columns
            )
        )

    # ========================================================
    # 5. PREPARE PROTOTYPE DATA
    # ========================================================

    print("\n" + "=" * 75)
    print("4. PREPARING FACILITY DATA")
    print("=" * 75)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = [
        "closing_inventory",
        "consumption",
        "latitude",
        "longitude",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_prototype_columns
    ).copy()

    print(
        f"Valid prototype records: {len(df):,}"
    )

    # ========================================================
    # 6. PREPARE PREDICTIONS
    # ========================================================

    print("\n" + "=" * 75)
    print("5. PREPARING ML SHORTAGE-RISK DATA")
    print("=" * 75)

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="coerce"
    )

    predictions["stockout_risk"] = pd.to_numeric(
        predictions["stockout_risk"],
        errors="coerce"
    )

    predictions = predictions.dropna(
        subset=[
            "facility_id",
            "date",
            "stockout_risk",
        ]
    ).copy()

    # Make sure the value behaves like a probability.

    predictions["stockout_risk"] = (
        predictions["stockout_risk"]
        .clip(
            lower=0.0,
            upper=1.0
        )
    )

    print(
        f"Valid prediction records: "
        f"{len(predictions):,}"
    )

    print(
        f"Prediction risk range: "
        f"{predictions['stockout_risk'].min():.4f}"
        f" - "
        f"{predictions['stockout_risk'].max():.4f}"
    )

    # ========================================================
    # 7. CURRENT FACILITY STATE
    # ========================================================

    print("\n" + "=" * 75)
    print("6. BUILDING CURRENT FACILITY STATE")
    print("=" * 75)

    latest_state = (
        df.sort_values("date")
        .groupby("facility_id")
        .tail(1)
        .copy()
    )

    print(
        f"Facilities with current state: "
        f"{len(latest_state):,}"
    )

    # ========================================================
    # 8. LATEST ML PREDICTION PER FACILITY
    # ========================================================

    latest_predictions = (
        predictions
        .sort_values("date")
        .groupby("facility_id")
        .tail(1)
        [
            [
                "facility_id",
                "date",
                "stockout_risk",
            ]
        ]
        .rename(
            columns={
                "date": "prediction_date",
            }
        )
    )

    print(
        f"Facilities with ML predictions: "
        f"{len(latest_predictions):,}"
    )

    # ========================================================
    # 9. MERGE STATE + PREDICTION
    # ========================================================

    facility_state = latest_state.merge(
        latest_predictions,
        on="facility_id",
        how="left"
    )

    facility_state[
        "stockout_risk"
    ] = facility_state[
        "stockout_risk"
    ].fillna(0.0)

    # ========================================================
    # 10. DEMAND BASELINE
    # ========================================================

    print("\n" + "=" * 75)
    print("7. CALCULATING INVENTORY COVERAGE")
    print("=" * 75)

    # Avoid division by zero.

    facility_state[
        "monthly_demand"
    ] = facility_state[
        "consumption"
    ].clip(
        lower=1
    )

    facility_state[
        "safety_adjusted_demand"
    ] = (
        facility_state[
            "monthly_demand"
        ]
        * (1.0 + SAFETY_MARGIN)
    )

    facility_state[
        "inventory_coverage_months"
    ] = (
        facility_state[
            "closing_inventory"
        ]
        /
        facility_state[
            "safety_adjusted_demand"
        ]
    )

    # ========================================================
    # 11. DONOR IDENTIFICATION
    # ========================================================

    print("\n" + "=" * 75)
    print("8. IDENTIFYING POTENTIAL DONORS")
    print("=" * 75)

    facility_state[
        "donor_safety_stock"
    ] = (
        facility_state[
            "safety_adjusted_demand"
        ]
        * DONOR_SAFETY_MONTHS
    )

    facility_state[
        "surplus_inventory"
    ] = (
        facility_state[
            "closing_inventory"
        ]
        -
        facility_state[
            "donor_safety_stock"
        ]
    ).clip(
        lower=0
    )

    facility_state[
        "potential_donor"
    ] = (
        (
            facility_state[
                "inventory_coverage_months"
            ]
            >= DONOR_COVERAGE_THRESHOLD
        )
        &
        (
            facility_state[
                "surplus_inventory"
            ]
            > 0
        )
    )

    donor_count = int(
        facility_state[
            "potential_donor"
        ].sum()
    )

    print(
        f"Potential donor facilities: "
        f"{donor_count:,}"
    )

    # ========================================================
    # 12. RECIPIENT IDENTIFICATION
    # ========================================================

    print("\n" + "=" * 75)
    print("9. IDENTIFYING SHORTAGE TARGETS")
    print("=" * 75)

    facility_state[
        "potential_recipient"
    ] = (
        facility_state[
            "stockout_risk"
        ]
        >= RECIPIENT_RISK_THRESHOLD
    )

    recipient_count = int(
        facility_state[
            "potential_recipient"
        ].sum()
    )

    print(
        f"Potential recipient facilities: "
        f"{recipient_count:,}"
    )

    # ========================================================
    # 13. BUILD DONOR AND RECIPIENT TABLES
    # ========================================================

    donors = facility_state[
        facility_state[
            "potential_donor"
        ]
    ].copy()

    recipients = facility_state[
        facility_state[
            "potential_recipient"
        ]
    ].copy()

    if len(donors) == 0:

        print(
            "\nNo potential donors found."
        )

        print(
            "Stage 8 cannot construct redistribution "
            "options with the current threshold."
        )

        return

    if len(recipients) == 0:

        print(
            "\nNo potential recipients found."
        )

        print(
            "Stage 8 cannot construct redistribution "
            "options with the current threshold."
        )

        return

    # ========================================================
    # 14. BUILD DONOR-RECIPIENT PAIRS
    # ========================================================

    print("\n" + "=" * 75)
    print("10. BUILDING GEOGRAPHIC REDISTRIBUTION OPTIONS")
    print("=" * 75)

    pairs = []

    for _, donor in donors.iterrows():

        for _, recipient in recipients.iterrows():

            if (
                donor["facility_id"]
                ==
                recipient["facility_id"]
            ):
                continue

            distance_km = haversine_distance(
                donor["latitude"],
                donor["longitude"],
                recipient["latitude"],
                recipient["longitude"]
            )

            if distance_km > MAX_DISTANCE_KM:
                continue

            pairs.append(
                {
                    "donor_facility_id":
                        donor["facility_id"],

                    "recipient_facility_id":
                        recipient["facility_id"],

                    "donor_district":
                        donor.get(
                            "district",
                            ""
                        ),

                    "recipient_district":
                        recipient.get(
                            "district",
                            ""
                        ),

                    "distance_km":
                        float(distance_km),

                    "donor_inventory":
                        float(
                            donor[
                                "closing_inventory"
                            ]
                        ),

                    "donor_surplus_inventory":
                        float(
                            donor[
                                "surplus_inventory"
                            ]
                        ),

                    "recipient_inventory":
                        float(
                            recipient[
                                "closing_inventory"
                            ]
                        ),

                    "recipient_monthly_demand":
                        float(
                            recipient[
                                "monthly_demand"
                            ]
                        ),

                    "recipient_stockout_probability":
                        float(
                            recipient[
                                "stockout_risk"
                            ]
                        ),
                }
            )

    pair_df = pd.DataFrame(
        pairs
    )

    print(
        f"Geographically feasible pairs: "
        f"{len(pair_df):,}"
    )

    if len(pair_df) == 0:

        print(
            "\nNo donor-recipient pairs found "
            f"within {MAX_DISTANCE_KM:.0f} km."
        )

        return

    # ========================================================
    # 15. RECIPIENT STOCKOUT TIME
    # ========================================================

    print("\n" + "=" * 75)
    print("11. ESTIMATING RECIPIENT STOCKOUT TIME")
    print("=" * 75)

    pair_df[
        "recipient_safety_adjusted_demand"
    ] = (
        pair_df[
            "recipient_monthly_demand"
        ]
        * (1.0 + SAFETY_MARGIN)
    )

    pair_df[
        "recipient_stockout_time_months"
    ] = (
        pair_df[
            "recipient_inventory"
        ]
        /
        pair_df[
            "recipient_safety_adjusted_demand"
        ]
    )

    # ========================================================
    # 16. MONTE CARLO TRANSIT FEASIBILITY
    # ========================================================

    print("\n" + "=" * 75)
    print("12. RUNNING MONTE CARLO TRANSIT SIMULATION")
    print("=" * 75)

    print(
        f"Simulations per pair: "
        f"{N_SIMULATIONS:,}"
    )

    print(
        "Transit distribution: LogNormal"
    )

    print(
        f"Median simulated transit time: "
        f"{TRANSIT_MEDIAN_MONTHS:.2f} months"
    )

    feasibility_probabilities = []

    for stockout_time in pair_df[
        "recipient_stockout_time_months"
    ]:

        probability = calculate_transit_probability(
            stockout_time,
            rng
        )

        feasibility_probabilities.append(
            probability
        )

    pair_df[
        "transit_feasibility_probability"
    ] = feasibility_probabilities

    # ========================================================
    # 17. DISTANCE FACTOR
    # ========================================================

    pair_df[
        "distance_factor"
    ] = (
        1.0
        -
        (
            pair_df[
                "distance_km"
            ]
            /
            MAX_DISTANCE_KM
        )
    ).clip(
        lower=0.0,
        upper=1.0
    )

    # ========================================================
    # 18. SURPLUS FACTOR
    # ========================================================

    pair_df[
        "surplus_factor"
    ] = (
        pair_df[
            "donor_surplus_inventory"
        ]
        /
        (
            pair_df[
                "donor_inventory"
            ]
            + 1.0
        )
    ).clip(
        lower=0.0,
        upper=1.0
    )

    # ========================================================
    # 19. RECIPIENT URGENCY
    # ========================================================

    pair_df[
        "recipient_urgency"
    ] = pair_df[
        "recipient_stockout_probability"
    ].clip(
        lower=0.0,
        upper=1.0
    )

    # ========================================================
    # 20. FINAL FEASIBILITY SCORE
    # ========================================================

    # This is a decision-support score.
    #
    # It is NOT a probability.
    #
    # Components:
    #
    # 50% transit feasibility
    # 20% geographic proximity
    # 15% donor surplus
    # 15% recipient urgency

    pair_df[
        "feasibility_score"
    ] = (
        0.50
        * pair_df[
            "transit_feasibility_probability"
        ]
        +
        0.20
        * pair_df[
            "distance_factor"
        ]
        +
        0.15
        * pair_df[
            "surplus_factor"
        ]
        +
        0.15
        * pair_df[
            "recipient_urgency"
        ]
    )

    # ========================================================
    # 21. FEASIBILITY CATEGORY
    # ========================================================

    pair_df[
        "feasibility_category"
    ] = pair_df[
        "feasibility_score"
    ].apply(
        classify_feasibility
    )

    pair_df = pair_df.sort_values(
        [
            "feasibility_score",
            "recipient_stockout_probability"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 22. RESULTS
    # ========================================================

    print("\n" + "=" * 75)
    print("13. RESULTS")
    print("=" * 75)

    print(
        f"Redistribution pairs evaluated: "
        f"{len(pair_df):,}"
    )

    print(
        "\nFeasibility categories:"
    )

    print(
        pair_df[
            "feasibility_category"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # 23. TOP REDISTRIBUTION OPTIONS
    # ========================================================

    print("\n" + "=" * 75)
    print("14. TOP REDISTRIBUTION OPTIONS")
    print("=" * 75)

    top_columns = [
        "donor_facility_id",
        "recipient_facility_id",
        "distance_km",
        "donor_surplus_inventory",
        "recipient_inventory",
        "recipient_stockout_probability",
        "recipient_stockout_time_months",
        "transit_feasibility_probability",
        "feasibility_score",
        "feasibility_category",
    ]

    print(
        pair_df[
            top_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 24. RECIPIENT SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("15. BUILDING RECIPIENT PRIORITY SUMMARY")
    print("=" * 75)

    # Best donor option for each recipient.

    recipient_summary = (
        pair_df
        .sort_values(
            "feasibility_score",
            ascending=False
        )
        .groupby(
            "recipient_facility_id"
        )
        .first()
        .reset_index()
    )

    donor_option_counts = (
        pair_df
        .groupby(
            "recipient_facility_id"
        )
        .size()
        .rename(
            "available_donor_options"
        )
        .reset_index()
    )

    recipient_summary = recipient_summary.merge(
        donor_option_counts,
        on="recipient_facility_id",
        how="left"
    )

    summary_columns = [
        "recipient_facility_id",
        "recipient_district",
        "recipient_stockout_probability",
        "recipient_inventory",
        "recipient_monthly_demand",
        "recipient_stockout_time_months",
        "donor_facility_id",
        "distance_km",
        "donor_surplus_inventory",
        "transit_feasibility_probability",
        "feasibility_score",
        "feasibility_category",
        "available_donor_options",
    ]

    recipient_summary = recipient_summary[
        summary_columns
    ]

    recipient_summary = recipient_summary.sort_values(
        [
            "recipient_stockout_probability",
            "feasibility_score"
        ],
        ascending=[
            False,
            False
        ]
    )

    print(
        recipient_summary
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 25. SAVE OUTPUTS
    # ========================================================

    print("\n" + "=" * 75)
    print("16. SAVING OUTPUTS")
    print("=" * 75)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pair_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    recipient_summary.to_csv(
        TARGET_SUMMARY_FILE,
        index=False
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Saved: {TARGET_SUMMARY_FILE}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 8 COMPLETE")
    print("=" * 75)

    print(
        "\nRedistribution feasibility signals are ready "
        "for the next stage."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Transit times and redistribution assumptions "
        "are simulated prototype assumptions."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()